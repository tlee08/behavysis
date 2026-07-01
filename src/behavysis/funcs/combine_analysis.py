"""Combine Analysis."""

from pathlib import Path

import polars as pl
from loguru import logger

from behavysis.constants import DF_IO_FORMAT, FBF
from behavysis.schemas import (
    ANALYSIS_SCHEMA,
    COMBINED_ANALYSIS_SCHEMA,
    read_df,
    write_df,
)


def combine_analysis(
    name: str,
    analysis_dir: Path,
    analysis_combined_fp: Path,
) -> None:
    """Combine analysis DataFrames across analysis types.

    Concatenates frame-by-frame analysis files from all analysis subdirectories,
    adding an ``analysis`` column to distinguish them.
    """
    analysis_subdir_ls = [
        i for i in analysis_dir.iterdir() if (analysis_dir / i).is_dir()
    ]
    if len(analysis_subdir_ls) == 0:
        logger.warning("no analysis fbf files made. Run `exp.analyse` first")
        return

    comb_df_ls = []
    for analysis_subdir in analysis_subdir_ls:
        fbf_fp = analysis_dir / analysis_subdir / FBF / f"{name}.{DF_IO_FORMAT}"
        if fbf_fp.is_file():
            df = read_df(fbf_fp, ANALYSIS_SCHEMA)
            df = df.with_columns(pl.lit(analysis_subdir.name).alias("analysis"))
            comb_df_ls.append(df)

    if not comb_df_ls:
        logger.warning("no analysis fbf files found")
        return

    combined = pl.concat(comb_df_ls)
    write_df(combined, analysis_combined_fp, COMBINED_ANALYSIS_SCHEMA)
