"""Analysis functions operating on Polars long-form keypoints DataFrames.

Functions have the following format:
    func(keypoints_fp, formatted_vid_fp, dst_dir, config_fp) -> None
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from pydantic import BaseModel, PositiveFloat

from behavysis.constants import BPTS_SIMBA, DF_IO_FORMAT, FBF
from behavysis.schemas import (
    ANALYSIS_SCHEMA,
    KEYPOINTS_SCHEMA,
    check_bpts_exist,
    get_indivs_bpts,
    read_df,
    summary_binned_quantitative,
    write_df,
)

from ._helper import _bodypart_avg_xy

if TYPE_CHECKING:
    from pathlib import Path

    from behavysis.models import ExperimentConfig, ExperimentMetadata


# ═══════════════════════════════════════════════════════════════════════════════
# Config Models
# ═══════════════════════════════════════════════════════════════════════════════


class SocialDistanceConfig(BaseModel):
    """SocialDistanceConfig."""

    smoothing_sec: PositiveFloat = 1.0
    bodyparts: list[str] = BPTS_SIMBA


# ═══════════════════════════════════════════════════════════════════════════════
# Functions
# ═══════════════════════════════════════════════════════════════════════════════


def social_distance(
    keypoints_fp: Path,
    formatted_vid_fp: Path,  # noqa: ARG001
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
    dst_dir: Path,
) -> None:
    """Determines the social distance between two individuals."""
    name = keypoints_fp.stem

    cfg = config.require_analyse().require(
        "social_distance",
        SocialDistanceConfig,
    )
    bpts = cfg.bodyparts
    smoothing_sec = cfg.smoothing_sec
    smoothing_frames = int(smoothing_sec * metadata.require_fps())

    keypoints_df = read_df(keypoints_fp, KEYPOINTS_SCHEMA)
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
            / metadata.require_px_per_mm()
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

    fbf_fp = dst_dir / FBF / f"{name}.{DF_IO_FORMAT}"
    write_df(analysis_df, fbf_fp, ANALYSIS_SCHEMA)

    summary_binned_quantitative(
        analysis_df,
        dst_dir,
        name,
        metadata.require_fps(),
        config.require_bins_sec_ls(),
        config.require_custom_bins_sec_ls(),
    )
