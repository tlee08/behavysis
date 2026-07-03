"""Analyse Behaviours."""

from pathlib import Path

import polars as pl

from behavysis.constants import DF_IO_FORMAT, FALSE_POS, FBF, UNSURE
from behavysis.models import ExperimentConfig, ExperimentMetadata
from behavysis.schemas import (
    ANALYSIS_SCHEMA,
    BEHAVIOUR_SCORED_BASE,
    read_df,
    write_df,
)
from behavysis.transforms.analysis import summary_binned_behaviour


def analyse_behaviour(
    behaviour_fp: Path,
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
    dst_dir: Path,
) -> None:
    """Takes a behaviour df and generates a summary and binned version of the data."""
    name = behaviour_fp.stem

    behaviour_df = read_df(behaviour_fp, BEHAVIOUR_SCORED_BASE)

    behaviour_df = behaviour_df.fill_null(0).with_columns(
        pl.when(pl.col("actual") == UNSURE)
        .then(FALSE_POS)
        .otherwise(pl.col("actual"))
        .alias("actual"),
    )

    drop_cols = ["pred"]
    keep_cols = [c for c in behaviour_df.columns if c not in drop_cols]

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

    analysis_df = analysis_df.with_columns(
        pl.col("value").fill_null(0),
    )

    fbf_fp = dst_dir / FBF / f"{name}.{DF_IO_FORMAT}"
    write_df(analysis_df, fbf_fp, ANALYSIS_SCHEMA)

    summary_binned_behaviour(
        analysis_df,
        dst_dir,
        name,
        metadata.require_fps(),
        config.require_analyse().bins_sec_ls,
        config.require_analyse().custom_bins_sec_ls(),
    )
