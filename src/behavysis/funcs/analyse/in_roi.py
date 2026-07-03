"""Analysis functions operating on Polars long-form keypoints DataFrames.

Functions have the following format:
    func(keypoints_fp, formatted_vid_fp, dst_dir, config_fp) -> None
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from pydantic import BaseModel, PositiveFloat

from behavysis.constants import BPTS_CORNERS, BPTS_SIMBA, DF_IO_FORMAT, FBF
from behavysis.funcs.analyse._helper import _bodypart_avg_xy
from behavysis.schemas import (
    ANALYSIS_SCHEMA,
    KEYPOINTS_SCHEMA,
    read_df,
    write_df,
)
from behavysis.transforms.analysis import summary_binned_behaviour
from behavysis.transforms.keypoint import check_bpts_exist, get_indivs_bpts

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

    from behavysis.models import ExperimentConfig, ExperimentMetadata

# ═══════════════════════════════════════════════════════════════════════════════
# Config Models
# ═══════════════════════════════════════════════════════════════════════════════


class InRoiConfig(BaseModel):
    """InRoiConfig."""

    roi_name: str = "in_my_roi"
    is_in: bool = True
    padding_mm: PositiveFloat = 0.0
    roi_corners: list[str] = BPTS_CORNERS
    bodyparts: list[str] = BPTS_SIMBA


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis functions
# ═══════════════════════════════════════════════════════════════════════════════


def in_roi(
    keypoints_fp: Path,
    formatted_vid_fp: Path,
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
    dst_dir: Path,
) -> None:
    """Determines frames where subject is inside ROI from average bpts."""
    name = keypoints_fp.stem

    cfg_ls = config.require_analyse().require_list("in_roi", InRoiConfig)

    keypoints_df = read_df(keypoints_fp, KEYPOINTS_SCHEMA)

    indivs, _ = get_indivs_bpts(keypoints_df)

    all_analysis_rows = []
    all_corners_rows = []
    roi_names = []

    for cfg in cfg_ls:
        roi_name = cfg.roi_name
        is_in = cfg.is_in
        bpts = cfg.bodyparts
        padding_mm = cfg.padding_mm
        roi_corners = cfg.roi_corners

        padding_px = padding_mm / metadata.require_px_per_mm()

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

    fbf_fp = dst_dir / FBF / f"{name}.{DF_IO_FORMAT}"
    write_df(analysis_df, fbf_fp, ANALYSIS_SCHEMA)

    # Scatter plot
    # Get video frame (150 frames in) or black frame if no video
    cap = cv2.VideoCapture(str(formatted_vid_fp))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 149)
    ret, frame_img = cap.read()
    cap.release()
    if not ret:
        frame_img = np.zeros(
            [metadata.require_height_px(), metadata.require_width_px(), 3],
        )
    # Make scatter plot
    plot_fp = dst_dir / "scatter_plot" / f"{name}.png"
    _make_location_scatterplot(analysis_df, corners_df, frame_img, plot_fp)

    summary_binned_behaviour(
        analysis_df,
        dst_dir,
        name,
        metadata.require_fps(),
        config.require_analyse().bins_sec_ls,
        config.require_analyse().custom_bins_sec_ls,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Funcs
# ═══════════════════════════════════════════════════════════════════════════════


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
    """Compute frame-by-frame movement distance for each individual."""
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
