"""Analysis functions operating on Polars long-form keypoints DataFrames."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from pydantic import BaseModel, PositiveFloat

from behavysis.constants import DF_IO_FORMAT, FBF
from behavysis.funcs.analyse._summary import summary_binned_quantitative
from behavysis.models import AnalysisResult
from behavysis.schemas import ANALYSIS_SCHEMA, write_df
from behavysis.transforms.keypoint import check_bpts_exist, get_indivs_bpts

from ._helper import _bodypart_avg_xy

if TYPE_CHECKING:
    from behavysis.models import ExperimentConfig, ExperimentMetadata


class SocialDistanceConfig(BaseModel):
    """SocialDistanceConfig."""

    bodyparts: list[str]
    smoothing_sec: PositiveFloat


def social_distance(
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
    *,
    keypoints_df: pl.DataFrame,
) -> list[AnalysisResult]:
    """Determines the social distance between two individuals."""
    name = metadata.require_name()

    cfg = config.require_analyse().require("social_distance", SocialDistanceConfig)
    bpts = cfg.bodyparts
    smoothing_sec = cfg.smoothing_sec
    smoothing_frames = int(smoothing_sec * metadata.require_fps())

    check_bpts_exist(keypoints_df, bpts)
    indivs, _ = get_indivs_bpts(keypoints_df)
    if len(indivs) < 2:  # noqa: PLR2004
        msg = "Social distance requires at least 2 individuals."
        raise ValueError(msg)

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
