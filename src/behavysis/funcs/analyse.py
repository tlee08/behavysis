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
from loguru import logger
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
    rows = []

    for indiv in indivs:
        avg = _bodypart_avg_xy(keypoints_df, indiv, bpts)

        # Smooth to reduce jitter
        x_smooth = (
            avg.select("x")
            .to_series()
            .rolling_mean(window_size=jitter_frames, min_samples=1, center=True)
        )
        y_smooth = (
            avg.select("y")
            .to_series()
            .rolling_mean(window_size=jitter_frames, min_samples=1, center=True)
        )

        # Frame-by-frame deltas
        delta_x = x_smooth.diff(null_behavior="ignore").fill_null(0)
        delta_y = y_smooth.diff(null_behavior="ignore").fill_null(0)
        delta_px = (delta_x.pow(2) + delta_y.pow(2)).sqrt()

        # Distance in mm
        dist_mm = delta_px / px_per_mm

        # Smoothed distance
        dist_mm_smoothed = dist_mm.rolling_mean(
            window_size=smoothing_frames,
            min_samples=1,
            center=True,
        )

        frames = avg.select("frame").to_series()

        for i in range(len(frames)):
            rows.append(
                {
                    "frame": int(frames[i]),
                    "individual": indiv,
                    "measure": "DistMM",
                    "value": float(dist_mm[i]),
                },
            )
            rows.append(
                {
                    "frame": int(frames[i]),
                    "individual": indiv,
                    "measure": "DistMMSmoothed",
                    "value": float(dist_mm_smoothed[i]),
                },
            )

    return pl.DataFrame(rows, schema=ANALYSIS_SCHEMA)


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

        # For each individual, determine in-roi status
        for indiv in indivs:
            avg = _bodypart_avg_xy(keypoints_df, indiv, bpts)
            frames = avg.select("frame").to_series().to_list()
            xs = avg.select("x").to_series().to_list()
            ys = avg.select("y").to_series().to_list()

            for f, px, py in zip(frames, xs, ys, strict=True):
                in_roi_val = _pt_in_roi(px, py, corners_i)
                if not is_in:
                    in_roi_val = not in_roi_val
                all_analysis_rows.append(
                    {
                        "frame": f,
                        "individual": indiv,
                        "measure": roi_name,
                        "value": float(int(in_roi_val)),
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
    formatted_vid_cap = cv2.VideoCapture(str(formatted_vid_fp))
    for _ in range(start_frame + 100):
        ret, frame_img = formatted_vid_cap.read()
        if not ret:
            logger.warning("Video shorter than start_frame")
            break
    formatted_vid_cap.release()

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

    # Convert distance to speed
    speed_rows = []
    for indiv in indivs:
        indiv_data = movement_df.filter(pl.col("individual") == indiv)
        for measure_name in ["DistMM", "DistMMSmoothed"]:
            dist_rows = indiv_data.filter(pl.col("measure") == measure_name)
            speed_rows.append(
                {
                    "frame": row["frame"],
                    "individual": indiv,
                    "measure": f"SpeedMMperSec{measure_name.replace('DistMM', '')}",
                    "value": row["value"] * analysis_config.fps,
                }
                for row in dist_rows.iter_rows(named=True)
            )

    analysis_df = pl.DataFrame(speed_rows, schema=ANALYSIS_SCHEMA)

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

    rows = [
        {
            "frame": row["frame"],
            "individual": pair_name,
            "measure": "DistMM",
            "value": row["DistMM"],
        }
        for row in dist.iter_rows(named=True)
    ]
    for _, row in enumerate(dist_smoothed.iter_rows(named=True)):
        rows.append(
            {
                "frame": row["frame"],
                "individual": pair_name,
                "measure": "DistMMSmoothed",
                "value": row["DistMMSmoothed"],
            },
        )

    analysis_df = pl.DataFrame(rows, schema=ANALYSIS_SCHEMA)

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

    all_rows = []

    for indiv in indivs:
        indiv_df = keypoints_df.filter(pl.col("individual") == indiv)
        frames = indiv_df.select("frame").unique().sort("frame").to_series()

        # For each bodypart, compute per-frame delta
        deltas_list = []
        for bpt in bpts:
            bpt_df = indiv_df.filter(pl.col("bodypart") == bpt).sort("frame")
            delta_x = (
                bpt_df.select("x").to_series().diff(null_behavior="drop").fill_null(0)
            )
            delta_y = (
                bpt_df.select("y").to_series().diff(null_behavior="drop").fill_null(0)
            )
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
        freezing_series = pl.Series(is_freezing.astype(int))
        bouts = vect2bouts(freezing_series == 1)
        for row in bouts.iter_rows(named=True):
            if row["dur"] < window_frames:
                freezing_series[row["start"] : row["stop"] + 1] = 0

        for i, f in enumerate(frames.to_list()):
            all_rows.append(
                {
                    "frame": f,
                    "individual": indiv,
                    "measure": "freezing",
                    "value": float(freezing_series[i]),
                },
            )

    analysis_df = pl.DataFrame(all_rows, schema=ANALYSIS_SCHEMA)

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
