"""Analysing behaviours: converts scored behaviour to analysis DataFrames."""

from pathlib import Path

import polars as pl

from behavysis.constants import (
    ACTUAL,
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
    """Takes a behaviour df and generates a summary and binned version of the data."""
    name = metadata.require_name()

    behaviour_df = behaviour_df.with_columns(
        pl.when(pl.col(ACTUAL) == TRUE_POS)
        .then(TRUE_POS)
        .otherwise(TRUE_NEG)
        .alias(ACTUAL),
    )

    id_vars = [FRAME, BEHAVIOUR]
    value_vars = [c for c in behaviour_df.columns if c not in id_vars]

    analysis_df = (
        behaviour_df.unpivot(
            index=id_vars, on=value_vars, variable_name=MEASURE, value_name=VALUE
        )
        .rename({BEHAVIOUR: INDIVIDUAL})
        .with_columns(pl.col(VALUE).fill_null(0).cast(pl.Float64))
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
