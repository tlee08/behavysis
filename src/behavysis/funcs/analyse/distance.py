"""Analysis functions operating on Polars long-form keypoints DataFrames."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from pydantic import BaseModel, PositiveFloat

from behavysis.constants import (
    DF_IO_FORMAT,
    FBF,
    FRAME,
    GROUP,
    MEASURE,
    VALUE,
    X,
    Y,
)
from behavysis.schemas import ANALYSIS_SCHEMA, write_df
from behavysis.transforms import bodypart_avg_xy, check_bpts_exist, get_indivs_bpts

from ._helper import AnalysisResult
from ._summary import summary_binned_quantitative

if TYPE_CHECKING:
    from behavysis.models import ExperimentConfig, ExperimentMetadata


class DistanceConfig(BaseModel):
    """DistanceConfig."""

    bodyparts: list[str]
    smoothing_sec: PositiveFloat


def distance(
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
    *,
    keypoints_df: pl.DataFrame,
) -> list[AnalysisResult]:
    """Determines the distance travelled by the subject in each frame."""
    name = metadata.require_name()

    cfg = config.require_analyse().require("distance", DistanceConfig)
    bpts = cfg.bodyparts
    smoothing_sec = cfg.smoothing_sec
    smoothing_frames = int(smoothing_sec * metadata.require_fps())

    check_bpts_exist(keypoints_df, bpts)
    indivs, _ = get_indivs_bpts(keypoints_df)

    analysis_df = _compute_movement(
        keypoints_df,
        bpts,
        indivs,
        metadata.require_px_per_mm(),
        smoothing_frames,
    )

    return [
        AnalysisResult(
            relative_path=Path(FBF) / f"{name}.{DF_IO_FORMAT}",
            result=analysis_df,
            save_func=lambda fp, obj: write_df(obj, fp, ANALYSIS_SCHEMA),
        ),
        *summary_binned_quantitative(
            analysis_df,
            name,
            metadata.require_fps(),
            config.require_analyse().bins_sec_ls,
            config.require_analyse().custom_bins_sec_ls,
        ),
    ]


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
        avg = bodypart_avg_xy(keypoints_df, indiv, bpts)

        dist = (
            avg.with_columns(
                (
                    avg.select(X)
                    .to_series()
                    .rolling_mean(window_size=jitter_frames, min_samples=1, center=True)
                    .diff()
                    .fill_null(0)
                    .alias("x_delta")
                ),
                (
                    avg.select(Y)
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

        dist_long = (
            dist.select(
                pl.col(FRAME),
                pl.lit(indiv).alias(GROUP),
                pl.col("DistMM"),
                pl.col("DistMMSmoothed"),
            )
            .unpivot(index=[FRAME, GROUP], variable_name=MEASURE, value_name=VALUE)
            .select(FRAME, MEASURE, GROUP, VALUE)
        )

        results.append(dist_long)

    if not results:
        return pl.DataFrame(schema=ANALYSIS_SCHEMA)

    return pl.concat(results)
