"""Analysis functions operating on Polars long-form keypoints DataFrames.

Functions have the following format:
    func(keypoints_fp, formatted_vid_fp, dst_dir, config_fp) -> None
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from pydantic import BaseModel, PositiveFloat

from behavysis.constants import BPTS_SIMBA, DF_IO_FORMAT, FBF
from behavysis.funcs.analyse._helper import _bodypart_avg_xy
from behavysis.schemas import (
    ANALYSIS_SCHEMA,
    KEYPOINTS_SCHEMA,
    read_df,
    write_df,
)
from behavysis.transforms.analysis import summary_binned_quantitative
from behavysis.transforms.keypoint import check_bpts_exist, get_indivs_bpts

if TYPE_CHECKING:
    from pathlib import Path

    from behavysis.models import ExperimentConfig, ExperimentMetadata

# ═══════════════════════════════════════════════════════════════════════════════
# Config Models
# ═══════════════════════════════════════════════════════════════════════════════


class SpeedConfig(BaseModel):
    """SpeedConfig."""

    smoothing_sec: PositiveFloat = 1.0
    bodyparts: list[str] = BPTS_SIMBA


# ═══════════════════════════════════════════════════════════════════════════════
# Functions
# ═══════════════════════════════════════════════════════════════════════════════


def speed(
    keypoints_fp: Path,
    formatted_vid_fp: Path,  # noqa: ARG001
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
    dst_dir: Path,
) -> None:
    """Determines the speed of the subject in each frame."""
    name = keypoints_fp.stem

    cfg = config.require_analyse().require("speed", SpeedConfig)
    bpts = cfg.bodyparts
    smoothing_sec = cfg.smoothing_sec
    smoothing_frames = int(smoothing_sec * metadata.require_fps())

    keypoints_df = read_df(keypoints_fp, KEYPOINTS_SCHEMA)
    check_bpts_exist(keypoints_df, bpts)
    indivs, _ = get_indivs_bpts(keypoints_df)

    movement_df = _compute_movement(
        keypoints_df,
        bpts,
        indivs,
        metadata.require_px_per_mm(),
        smoothing_frames,
    )

    # Convert distance to speed via Polars vectorized operations
    analysis_df = movement_df.select(
        pl.col("frame"),
        pl.col("individual"),
        pl.col("measure").str.replace("DistMM", "SpeedMMperSec").alias("measure"),
        (pl.col("value") * metadata.require_fps()).alias("value"),
    )

    fbf_fp = dst_dir / FBF / f"{name}.{DF_IO_FORMAT}"
    write_df(analysis_df, fbf_fp, ANALYSIS_SCHEMA)

    summary_binned_quantitative(
        analysis_df,
        dst_dir,
        name,
        metadata.require_fps(),
        config.require_analyse().bins_sec_ls,
        config.require_analyse().custom_bins_sec_ls,
    )


def distance(
    keypoints_fp: Path,
    formatted_vid_fp: Path,  # noqa: ARG001
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
    dst_dir: Path,
) -> None:
    """Determines the distance travelled by the subject in each frame."""
    name = keypoints_fp.stem

    cfg = config.require_analyse().require("speed", SpeedConfig)
    bpts = cfg.bodyparts
    smoothing_sec = cfg.smoothing_sec
    smoothing_frames = int(smoothing_sec * metadata.require_fps())

    keypoints_df = read_df(keypoints_fp, KEYPOINTS_SCHEMA)
    check_bpts_exist(keypoints_df, bpts)
    indivs, _ = get_indivs_bpts(keypoints_df)

    analysis_df = _compute_movement(
        keypoints_df,
        bpts,
        indivs,
        metadata.require_px_per_mm(),
        smoothing_frames,
    )

    fbf_fp = dst_dir / FBF / f"{name}.{DF_IO_FORMAT}"
    write_df(analysis_df, fbf_fp, ANALYSIS_SCHEMA)

    summary_binned_quantitative(
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
