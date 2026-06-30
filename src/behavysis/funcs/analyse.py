"""Analysis functions operating on Polars long-form keypoints DataFrames.

Functions have the following format:
    func(keypoints_fp, formatted_vid_fp, dst_dir, config_fp) -> None
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt

from behavysis.constants import DF_IO_FORMAT, FBF
from behavysis.models import ExperimentConfig
from behavysis.schemas import (
    ANALYSIS_SCHEMA,
    KEYPOINTS_SCHEMA,
    check_bpts_exist,
    get_indivs_bpts,
    read_df,
    summary_binned_behaviour,
    summary_binned_quantitative,
    vect2bouts,
    write_df,
)

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes


class AnalyseFunc(Protocol):
    """Protocol for analyse functions."""

    def __call__(
        self,
        keypoints_fp: Path,
        formatted_vid_fp: Path,
        dst_dir: Path,
        config_fp: Path,
    ) -> None:
        """Protocol for analyse functions."""


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _bodypart_avg_xy(
    df: pl.DataFrame,
    indiv: str,
    bpts: list[str],
) -> pl.DataFrame:
    """Average x and y coordinates across bodyparts per frame for an individual.

    Returns DataFrame with frame, x_mean, y_mean.
    """
    return (
        df.filter(
            pl.col("individual") == indiv,
            pl.col("bodypart").is_in(bpts),
        )
        .group_by("frame")
        .agg([pl.col("x").mean().alias("x"), pl.col("y").mean().alias("y")])
        .sort("frame")
    )


def _pt_in_roi(pt_x: float, pt_y: float, corners_df: pl.DataFrame) -> bool:
    """Check if point is inside polygon using ray casting algorithm."""
    crossings = 0
    n = corners_df.height
    for i in range(n):
        c1 = corners_df.row(i, named=True)
        c2 = corners_df.row((i + 1) % n, named=True)
        y_between = (c1["y"] > pt_y) != (c2["y"] > pt_y)
        if y_between:
            x_int = (c2["x"] - c1["x"]) * (pt_y - c1["y"]) / (c2["y"] - c1["y"]) + c1[
                "x"
            ]
            if pt_x < x_int:
                crossings += 1
    return crossings % 2 == 1


def _pts_in_roi(
    px_arr: np.ndarray,
    py_arr: np.ndarray,
    corners_df: pl.DataFrame,
) -> np.ndarray:
    """Vectorized point-in-polygon using ray casting on numpy arrays."""
    n = corners_df.height
    cx = corners_df.select("x").to_numpy()
    cy = corners_df.select("y").to_numpy()

    crossings = np.zeros(len(px_arr), dtype=np.int32)
    for i in range(n):
        c1_x, c1_y = cx[i], cy[i]
        c2_x, c2_y = cx[(i + 1) % n], cy[(i + 1) % n]
        y_between = (c1_y > py_arr) != (c2_y > py_arr)
        if y_between.any():
            x_int = (c2_x - c1_x) * (py_arr[y_between] - c1_y) / (c2_y - c1_y) + c1_x
            crossings[y_between] ^= (px_arr[y_between] < x_int).astype(np.int32)
    return (crossings % 2) == 1


def _compute_movement(
    keypoints_df: pl.DataFrame,
    bpts: list[str],
    indivs: list[str],
    px_per_mm: float,
    smoothing_frames: int,
) -> pl.DataFrame:
    """Compute frame-by-frame movement distance for each individual.

    Returns DataFrame in ANALYSIS_SCHEMA format.
    """
    jitter_frames = 3
    results = []

    for indiv in indivs:
        avg = _bodypart_avg_xy(keypoints_df, indiv, bpts)

        dist = (
            avg.with_columns(
                (
                    avg.select("x")
                    .to_series()
                    .rolling_mean(window_size=jitter_frames, min_samples=1, center=True)
                    .diff()
                    .fill_null(0)
                    .alias("x_delta")
                ),
                (
                    avg.select("y")
                    .to_series()
                    .rolling_mean(window_size=jitter_frames, min_samples=1, center=True)
                    .diff()
                    .fill_null(0)
                    .alias("y_delta")
                ),
            )
            .with_columns(
                (
                    (pl.col("x_delta").pow(2) + pl.col("y_delta").pow(2)).sqrt()
                    / px_per_mm
                ).alias("DistMM"),
            )
            .with_columns(
                pl.col("DistMM")
                .rolling_mean(window_size=smoothing_frames, min_samples=1, center=True)
                .alias("DistMMSmoothed"),
            )
        )

        dist_long = dist.select(
            pl.col("frame"),
            pl.lit(indiv).alias("individual"),
            pl.col("DistMM"),
            pl.col("DistMMSmoothed"),
        ).unpivot(
            index=["frame", "individual"],
            variable_name="measure",
            value_name="value",
        )

        results.append(dist_long)

    if not results:
        return pl.DataFrame(schema=ANALYSIS_SCHEMA)

    return pl.concat(results)


def _make_location_scatterplot(
    scatter_df: pl.DataFrame,
    corners_df: pl.DataFrame,
    frame: np.ndarray,
    dst_fp: Path,
) -> None:
    """Make location scatterplot from Polars long-form scatter data."""
    # scatter_df is in ANALYSIS_SCHEMA with x and y as measure values
    indivs = (
        scatter_df.select("individual")
        .unique()
        .sort("individual")
        .to_series()
        .to_list()
    )
    measures = scatter_df.select("measure").unique().to_series().to_list()
    roi_ls = [m for m in measures if m not in ["x", "y"]]

    ax_size = 5
    nrows = max(len(roi_ls), 1)
    ncols = max(len(indivs), 1)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ax_size * ncols, ax_size * nrows),
    )
    axes = np.atleast_2d(np.asarray(axes)).reshape(nrows, ncols)

    for i, roi in enumerate(roi_ls):
        for j, indiv in enumerate(indivs):
            ax: Axes = axes[i, j]
            ax.imshow(frame, alpha=0.5)

            # Plot x-y scatter for this individual, colored by ROI
            indiv_data = scatter_df.filter(pl.col("individual") == indiv)
            plot_data = indiv_data.pivot(
                index="frame",
                on="measure",
                values="value",
            ).to_pandas()

            if (
                roi in plot_data.columns
                and "x" in plot_data.columns
                and "y" in plot_data.columns
            ):
                sns.scatterplot(
                    data=plot_data,
                    x="x",
                    y="y",
                    hue=roi,
                    palette={0: "orange", 1: "green"},
                    alpha=0.3,
                    linewidth=0,
                    marker=".",
                    s=5,
                    legend=False,
                    ax=ax,
                )

            # ROI polygon
            if corners_df is not None and "roi" in corners_df.columns:
                roi_corners = corners_df.filter(pl.col("roi") == roi)
                if roi_corners.height > 0:
                    corners_pd = roi_corners.to_pandas()
                    sns.lineplot(
                        data=corners_pd,
                        x="x",
                        y="y",
                        linewidth=1,
                        marker="+",
                        markeredgecolor=(1, 0, 0),
                        markeredgewidth=2,
                        markersize=5,
                        estimator=None,
                        sort=False,
                        legend=False,
                        ax=ax,
                    )

            ax.set_title(f"{roi} - {indiv}")
            ax.set_aspect("equal")

    dst_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst_fp)
    fig.clf()


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis functions
# ═══════════════════════════════════════════════════════════════════════════════


def in_roi(
    keypoints_fp: Path,
    formatted_vid_fp: Path,
    dst_dir: Path,
    config_fp: Path,
) -> None:
    """Determines frames where subject is inside ROI from average bpts."""
    name = keypoints_fp.stem
    dst_subdir = dst_dir / "in_roi"

    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    analysis_config = config.get_analysis_config()
    start_frame = config.auto.start_frame
    config_filt_ls = config.user.analyse.in_roi

    keypoints_df = read_df(keypoints_fp, KEYPOINTS_SCHEMA)
    assert keypoints_df.height > 0, "No frames in keypoints_df."

    indivs, _ = get_indivs_bpts(keypoints_df)

    all_analysis_rows = []
    all_corners_rows = []
    roi_names = []

    for config_filt in config_filt_ls:
        roi_name = config.get_ref(config_filt.roi_name)
        is_in = config.get_ref(config_filt.is_in)
        bpts = config.get_ref(config_filt.bodyparts)
        padding_mm = config.get_ref(config_filt.padding_mm)
        roi_corners = config.get_ref(config_filt.roi_corners)

        padding_px = padding_mm / analysis_config.px_per_mm

        check_bpts_exist(keypoints_df, bpts)
        check_bpts_exist(keypoints_df, roi_corners)

        # Average corner coordinates (assumed stationary)
        corners_rows = []
        for pt in roi_corners:
            avg = keypoints_df.filter(pl.col("bodypart") == pt).select(
                pl.col("x").mean().alias("x"),
                pl.col("y").mean().alias("y"),
            )
            corners_rows.append(avg)

        corners_i = pl.concat(corners_rows)
        corners_i = (
            corners_i.drop("likelihood")
            if "likelihood" in corners_i.columns
            else corners_i
        )

        # Adjust corners by padding
        roi_center = corners_i.select(pl.col("x").mean(), pl.col("y").mean())
        cx, cy = roi_center.row(0)

        adjusted = []
        for row in corners_i.iter_rows(named=True):
            theta = np.arctan2(row["y"] - cy, row["x"] - cx)
            adjusted.append(
                {
                    "x": row["x"] + padding_px * np.cos(theta),
                    "y": row["y"] + padding_px * np.sin(theta),
                },
            )
        corners_i = pl.DataFrame(adjusted)

        # For each individual, determine in-roi status (vectorized)
        for indiv in indivs:
            avg = _bodypart_avg_xy(keypoints_df, indiv, bpts)
            frames = avg.select("frame").to_series().to_numpy()
            xs = avg.select("x").to_series().to_numpy()
            ys = avg.select("y").to_series().to_numpy()

            in_roi_mask = _pts_in_roi(xs, ys, corners_i)
            if not is_in:
                in_roi_mask = ~in_roi_mask

            for f, val in zip(frames, in_roi_mask, strict=True):
                all_analysis_rows.append(
                    {
                        "frame": int(f),
                        "individual": indiv,
                        "measure": roi_name,
                        "value": float(val),
                    },
                )

        # Store corner positions for scatter plot
        all_corners_rows.extend(
            {
                "roi": roi_name,
                "x": row["x"],
                "y": row["y"],
            }
            for row in corners_i.iter_rows(named=True)
        )
        roi_names.append(roi_name)

    analysis_df = pl.DataFrame(all_analysis_rows, schema=ANALYSIS_SCHEMA)
    corners_df = pl.DataFrame(all_corners_rows)

    fbf_fp = dst_subdir / FBF / f"{name}.{DF_IO_FORMAT}"
    write_df(analysis_df, fbf_fp, ANALYSIS_SCHEMA)

    # Scatter plot
    # Get video frame (150 frames in) or black frame if no video
    formatted_vid_cap = cv2.VideoCapture(str(formatted_vid_fp))
    formatted_vid_cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
    ret, frame_img = formatted_vid_cap.read()
    if not ret:
        frame_img = np.zeros([analysis_config.height_px, analysis_config.width_px, 3])
    formatted_vid_cap.release()
    # Make scatter plot
    plot_fp = dst_subdir / "scatter_plot" / f"{name}.png"
    _make_location_scatterplot(analysis_df, corners_df, frame_img, plot_fp)

    summary_binned_behaviour(
        analysis_df,
        dst_subdir,
        name,
        analysis_config.fps,
        analysis_config.bins_sec,
        analysis_config.custom_bins_sec,
    )


def speed(
    keypoints_fp: Path,
    formatted_vid_fp: Path,  # noqa: ARG001
    dst_dir: Path,
    config_fp: Path,
) -> None:
    """Determines the speed of the subject in each frame."""
    name = keypoints_fp.stem
    dst_subdir = dst_dir / "speed"

    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    analysis_config = config.get_analysis_config()
    config_filt = config.user.analyse.speed
    bpts = config.get_ref(config_filt.bodyparts)
    smoothing_sec = config.get_ref(config_filt.smoothing_sec)
    smoothing_frames = int(smoothing_sec * analysis_config.fps)

    keypoints_df = read_df(keypoints_fp, KEYPOINTS_SCHEMA)
    assert keypoints_df.height > 0, "No frames in keypoints_df."
    check_bpts_exist(keypoints_df, bpts)
    indivs, _ = get_indivs_bpts(keypoints_df)

    movement_df = _compute_movement(
        keypoints_df,
        bpts,
        indivs,
        analysis_config.px_per_mm,
        smoothing_frames,
    )

    # Convert distance to speed via Polars vectorized operations
    analysis_df = movement_df.select(
        pl.col("frame"),
        pl.col("individual"),
        pl.col("measure").str.replace("DistMM", "SpeedMMperSec").alias("measure"),
        (pl.col("value") * analysis_config.fps).alias("value"),
    )

    fbf_fp = dst_subdir / FBF / f"{name}.{DF_IO_FORMAT}"
    write_df(analysis_df, fbf_fp, ANALYSIS_SCHEMA)

    summary_binned_quantitative(
        analysis_df,
        dst_subdir,
        name,
        analysis_config.fps,
        analysis_config.bins_sec,
        analysis_config.custom_bins_sec,
    )


def distance(
    keypoints_fp: Path,
    formatted_vid_fp: Path,  # noqa: ARG001
    dst_dir: Path,
    config_fp: Path,
) -> None:
    """Determines the distance travelled by the subject in each frame."""
    name = keypoints_fp.stem
    dst_subdir = dst_dir / "distance"

    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    analysis_config = config.get_analysis_config()
    config_filt = config.user.analyse.speed
    bpts = config.get_ref(config_filt.bodyparts)
    smoothing_sec = config.get_ref(config_filt.smoothing_sec)
    smoothing_frames = int(smoothing_sec * analysis_config.fps)

    keypoints_df = read_df(keypoints_fp, KEYPOINTS_SCHEMA)
    assert keypoints_df.height > 0, "No frames in keypoints_df."
    check_bpts_exist(keypoints_df, bpts)
    indivs, _ = get_indivs_bpts(keypoints_df)

    analysis_df = _compute_movement(
        keypoints_df,
        bpts,
        indivs,
        analysis_config.px_per_mm,
        smoothing_frames,
    )

    fbf_fp = dst_subdir / FBF / f"{name}.{DF_IO_FORMAT}"
    write_df(analysis_df, fbf_fp, ANALYSIS_SCHEMA)

    summary_binned_quantitative(
        analysis_df,
        dst_subdir,
        name,
        analysis_config.fps,
        analysis_config.bins_sec,
        analysis_config.custom_bins_sec,
    )


def social_distance(
    keypoints_fp: Path,
    formatted_vid_fp: Path,  # noqa: ARG001
    dst_dir: Path,
    config_fp: Path,
) -> None:
    """Determines the social distance between two individuals."""
    name = keypoints_fp.stem
    dst_subdir = dst_dir / "social_distance"

    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    analysis_config = config.get_analysis_config()
    config_filt = config.user.analyse.social_distance
    bpts = config.get_ref(config_filt.bodyparts)
    smoothing_sec = config.get_ref(config_filt.smoothing_sec)
    smoothing_frames = int(smoothing_sec * analysis_config.fps)

    keypoints_df = read_df(keypoints_fp, KEYPOINTS_SCHEMA)
    assert keypoints_df.height > 0, "No frames in keypoints_df."
    check_bpts_exist(keypoints_df, bpts)
    indivs, _ = get_indivs_bpts(keypoints_df)
    assert len(indivs) >= 2, "Social distance requires at least 2 individuals."

    indiv_a, indiv_b = indivs[0], indivs[1]
    pair_name = f"{indiv_a}_{indiv_b}"

    avg_a = _bodypart_avg_xy(keypoints_df, indiv_a, bpts)
    avg_b = _bodypart_avg_xy(keypoints_df, indiv_b, bpts)

    dist = avg_a.join(avg_b, on="frame", suffix="_b").with_columns(
        (
            (
                (pl.col("x") - pl.col("x_b")).pow(2)
                + (pl.col("y") - pl.col("y_b")).pow(2)
            ).sqrt()
            / analysis_config.px_per_mm
        ).alias("DistMM"),
    )

    dist_smoothed = dist.select("frame").with_columns(
        dist.select("DistMM")
        .to_series()
        .rolling_mean(
            window_size=smoothing_frames,
            min_samples=1,
            center=True,
        )
        .alias("DistMMSmoothed"),
    )

    analysis_df = (
        dist.join(dist_smoothed, on="frame")
        .select(
            pl.col("frame"),
            pl.lit(pair_name).alias("individual"),
            pl.col("DistMM"),
            pl.col("DistMMSmoothed"),
        )
        .unpivot(
            index=["frame", "individual"],
            variable_name="measure",
            value_name="value",
        )
    )

    fbf_fp = dst_subdir / FBF / f"{name}.{DF_IO_FORMAT}"
    write_df(analysis_df, fbf_fp, ANALYSIS_SCHEMA)

    summary_binned_quantitative(
        analysis_df,
        dst_subdir,
        name,
        analysis_config.fps,
        analysis_config.bins_sec,
        analysis_config.custom_bins_sec,
    )


def freezing(
    keypoints_fp: Path,
    formatted_vid_fp: Path,  # noqa: ARG001
    dst_dir: Path,
    config_fp: Path,
) -> None:
    """Determines frames where the subject is frozen (movement below threshold)."""
    name = keypoints_fp.stem
    dst_subdir = dst_dir / "freezing"

    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    analysis_config = config.get_analysis_config()
    config_filt = config.user.analyse.freezing
    bpts = config.get_ref(config_filt.bodyparts)
    thresh_mm = config.get_ref(config_filt.thresh_mm)
    smoothing_sec = config.get_ref(config_filt.smoothing_sec)
    window_sec = config.get_ref(config_filt.window_sec)

    thresh_px = thresh_mm / analysis_config.px_per_mm
    smoothing_frames = int(smoothing_sec * analysis_config.fps)
    window_frames = int(np.round(analysis_config.fps * window_sec))

    keypoints_df = read_df(keypoints_fp, KEYPOINTS_SCHEMA)
    assert keypoints_df.height > 0, "No frames in keypoints_df."
    check_bpts_exist(keypoints_df, bpts)
    indivs, _ = get_indivs_bpts(keypoints_df)

    all_dfs = []

    for indiv in indivs:
        indiv_df = keypoints_df.filter(pl.col("individual") == indiv)
        frames = indiv_df.select("frame").unique().sort("frame").to_series()

        # For each bodypart, compute per-frame delta
        deltas_list = []
        for bpt in bpts:
            bpt_df = indiv_df.filter(pl.col("bodypart") == bpt).sort("frame")
            delta_x = bpt_df.select("x").to_series().diff().fill_null(0)
            delta_y = bpt_df.select("y").to_series().diff().fill_null(0)
            delta = (delta_x.pow(2) + delta_y.pow(2)).sqrt()
            smoothed = delta.rolling_mean(
                window_size=smoothing_frames,
                min_samples=1,
                center=True,
            )
            deltas_list.append(smoothed.to_list())

        # Freezing if ALL bodyparts are below threshold
        n = len(frames)
        is_freezing = np.ones(n, dtype=bool)
        for deltas in deltas_list:
            is_freezing &= np.array(deltas[:n]) < thresh_px

        # Filter out short freezing bouts
        freezing_np = is_freezing.astype(np.int32)
        bouts = vect2bouts(pl.Series(freezing_np) == 1)
        for row in bouts.iter_rows(named=True):
            if row["dur"] < window_frames:
                freezing_np[row["start"] : row["stop"] + 1] = 0

        all_dfs.append(
            pl.DataFrame(
                {
                    "frame": frames,
                    "individual": indiv,
                    "measure": "freezing",
                    "value": freezing_np.astype(np.float64),
                },
                schema=ANALYSIS_SCHEMA,
            ),
        )

    analysis_df = pl.concat(all_dfs)

    fbf_fp = dst_subdir / FBF / f"{name}.{DF_IO_FORMAT}"
    write_df(analysis_df, fbf_fp, ANALYSIS_SCHEMA)

    summary_binned_behaviour(
        analysis_df,
        dst_subdir,
        name,
        analysis_config.fps,
        analysis_config.bins_sec,
        analysis_config.custom_bins_sec,
    )
