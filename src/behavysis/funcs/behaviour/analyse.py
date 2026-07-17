"""Analyse Behaviours."""

from pathlib import Path

import polars as pl

from behavysis.constants import ACTUAL, DF_IO_FORMAT, FALSE_POS, FBF, UNSURE
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

    # Set UNSURE vals to FALSE_POS
    behaviour_df = behaviour_df.fill_null(0).with_columns(
        pl.when(pl.col(ACTUAL) == UNSURE)
        .then(FALSE_POS)
        .otherwise(pl.col(ACTUAL))
        .alias(ACTUAL),
    )

    id_vars = ["frame", "behaviour"]
    value_vars = [c for c in behaviour_df.columns if c not in id_vars]

    analysis_df = behaviour_df.unpivot(
        index=id_vars,
        on=value_vars,
        variable_name="measure",
        value_name="value",
    ).rename({"behaviour": "individual"})

    analysis_df = analysis_df.with_columns(
        pl.col("value").fill_null(0),
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
