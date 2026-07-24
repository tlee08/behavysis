"""Analyse Behaviours."""

from pathlib import Path

import polars as pl

from behavysis.constants import ACTUAL, DF_IO_FORMAT, FBF, TRUE_NEG, TRUE_POS
from behavysis.models import AnalysisResult, ExperimentConfig, ExperimentMetadata
from behavysis.schemas import ANALYSIS_SCHEMA, write_df
from behavysis.transforms.analysis import summary_binned_behaviour


def analyse_behaviour(
    behaviour_df: pl.DataFrame,
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
) -> list[AnalysisResult]:
    """Takes a behaviour df and generates a summary and binned version of the data."""
    name = metadata.require_name()

    # Set only TRUE_POS as 1 (TRUE_POS). Everything else set to 0 (TRUE_NEG)
    behaviour_df = behaviour_df.with_columns(
        pl.when(pl.col(ACTUAL) == TRUE_POS)
        .then(TRUE_POS)
        .otherwise(TRUE_NEG)
        .alias(ACTUAL),
    )

    id_vars = ["frame", "behaviour"]
    value_vars = [c for c in behaviour_df.columns if c not in id_vars]

    analysis_df = (
        behaviour_df.unpivot(
            index=id_vars,
            on=value_vars,
            variable_name="measure",
            value_name="value",
        )
        .rename({"behaviour": "individual"})  # Just to fit analysis schema
        .with_columns(
            pl.col("value").fill_null(0).cast(pl.Float64),
        )
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
