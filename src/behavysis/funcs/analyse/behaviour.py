"""Analysing behaviours: converts scored behaviour to analysis DataFrames."""

from pathlib import Path

import polars as pl

from behavysis.constants import (
    BEHAVIOUR,
    DF_IO_FORMAT,
    FBF,
    FRAME,
    INDIVIDUAL,
    MEASURE,
    TRUE_NEG,
    TRUE_POS,
    VALUE,
)
from behavysis.models import ExperimentConfig, ExperimentMetadata
from behavysis.schemas import ANALYSIS_SCHEMA, write_df

from ._helper import AnalysisResult
from ._summary import summary_binned_behaviour


def analyse_behaviour(
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
    *,
    behaviour_df: pl.DataFrame,
) -> list[AnalysisResult]:
    """Takes a wide-format behaviour df and generates summary + binned analysis."""
    name = metadata.require_name()

    rows: list[pl.DataFrame] = []
    for behaviour, ref in config.require_classify_behaviour().items():
        behaviour_vals = behaviour_df.select(
            FRAME,
            pl.when(pl.col(behaviour) == TRUE_POS)
            .then(TRUE_POS)
            .otherwise(TRUE_NEG)
            .alias(VALUE),
        ).with_columns(pl.lit(behaviour).alias(MEASURE))
        rows.append(behaviour_vals)

        for sub in ref.sub_behaviour:
            sub_vals = behaviour_df.select(
                FRAME,
                pl.when(pl.col(sub) == TRUE_POS)
                .then(TRUE_POS)
                .otherwise(TRUE_NEG)
                .alias(VALUE),
            ).with_columns(pl.lit(sub).alias(MEASURE))
            rows.append(sub_vals)

    analysis_df = (
        pl.concat(rows)
        .with_columns(pl.lit(BEHAVIOUR).alias(INDIVIDUAL))
        .select(FRAME, INDIVIDUAL, MEASURE, VALUE)
        .with_columns(pl.col(VALUE).cast(pl.Float64))
    )

    return [
        AnalysisResult(
            relative_path=Path(FBF) / f"{name}.{DF_IO_FORMAT}",
            result=analysis_df,
            save_func=lambda fp, obj: write_df(obj, fp, ANALYSIS_SCHEMA),
        ),
        *summary_binned_behaviour(
            analysis_df,
            name,
            metadata.require_fps(),
            config.require_analyse().bins_sec_ls,
            config.require_analyse().custom_bins_sec_ls,
        ),
    ]
