"""Analyse Behaviours."""

from pathlib import Path

import polars as pl

from behavysis.constants import DF_IO_FORMAT, FALSE_POS, FBF, UNSURE
from behavysis.models import ExperimentConfig
from behavysis.schemas import (
    ANALYSIS_SCHEMA,
    BEHAVIOUR_SCORED_BASE,
    read_df,
    summary_binned_behaviour,
    write_df,
)


def analyse_behaviour(
    behaviour_fp: Path,
    dst_dir: Path,
    config_fp: Path,
) -> None:
    """Takes a behaviour df and generates a summary and binned version of the data."""
    name = behaviour_fp.stem
    dst_subdir = dst_dir / "analyse_behaviour"

    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    analysis_config = config.get_analysis_config()

    # Read Polars long-form scored behaviour DataFrame
    behaviour_df = read_df(behaviour_fp, BEHAVIOUR_SCORED_BASE)

    # Fill nulls, replace UNSURE with FALSE_POS
    behaviour_df = behaviour_df.fill_null(0).with_columns(
        pl.when(pl.col("actual") == UNSURE)
        .then(FALSE_POS)
        .otherwise(pl.col("actual"))
        .alias("actual"),
    )

    # Drop pred column, keep actual and user_defined columns
    drop_cols = ["pred"]
    keep_cols = [c for c in behaviour_df.columns if c not in drop_cols]

    # Melt to ANALYSIS_SCHEMA: (frame, individual, measure, value)
    # individual = behaviour name, measure = outcome column name
    id_vars = ["frame", "behaviour"]
    value_vars = [c for c in keep_cols if c not in id_vars]

    analysis_df = (
        behaviour_df.select(id_vars + value_vars)
        .unpivot(
            index=id_vars,
            on=value_vars,
            variable_name="measure",
            value_name="value",
        )
        .rename({"behaviour": "individual"})
    )

    # Fill null values with 0
    analysis_df = analysis_df.with_columns(
        pl.col("value").fill_null(0),
    )

    # Write frame-by-frame analysis
    fbf_fp = dst_subdir / FBF / f"{name}.{DF_IO_FORMAT}"
    write_df(analysis_df, fbf_fp, ANALYSIS_SCHEMA)

    # Generate summary and binned dataframes
    summary_binned_behaviour(
        analysis_df,
        dst_subdir,
        name,
        analysis_config.fps,
        analysis_config.bins_sec,
        analysis_config.custom_bins_sec,
    )
