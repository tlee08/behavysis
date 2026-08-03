"""Keypoints utility functions operating on Polars long-form DataFrames.

All functions operate on DataFrames conforming to
``KEYPOINTS_SCHEMA``: (frame, individual, bodypart, x, y, likelihood).
"""

from pathlib import Path

import pandas as pd
import polars as pl

from behavysis.constants import (
    BODYPART,
    FRAME,
    INDIVIDUAL,
    LIKELIHOOD,
    PROCESSED,
    SINGLE,
    X,
    Y,
)
from behavysis.schemas import KEYPOINTS_SCHEMA, write_df


def check_bpts_exist(df: pl.DataFrame, bodyparts: list[str]) -> None:
    """Check that all requested bodyparts exist in the keypoints DataFrame.

    Raises ValueError with available bodyparts if any are missing.
    """
    available_set = set(df.select(BODYPART).to_series().unique().sort().to_list())
    missing = [b for b in bodyparts if b not in available_set]
    if missing:
        max_missing = 5
        avail_list = sorted(available_set)[:max_missing]
        suffix = "..." if len(available_set) > max_missing else ""
        msg = (
            f"Bodyparts not found in keypoints data: {missing}\n"
            f"  Available: {', '.join(avail_list)}{suffix}\n"
            f"  Check your config file's bodyparts list."
        )
        raise ValueError(msg)


def get_indivs_bpts(df: pl.DataFrame) -> tuple[list[str], list[str]]:
    """Get individuals and bodyparts excluding special markers.

    Filters out ``single`` and ``processed`` individuals.
    """
    filtered = (
        df.filter(
            ~pl.col(INDIVIDUAL).is_in([PROCESSED, SINGLE]),
        )
        .select([INDIVIDUAL, BODYPART])
        .unique()
        .sort([INDIVIDUAL, BODYPART])
    )

    individuals = (
        filtered.select(INDIVIDUAL).unique().sort(INDIVIDUAL).to_series().to_list()
    )
    bodyparts = filtered.select(BODYPART).unique().sort(BODYPART).to_series().to_list()
    return individuals, bodyparts


def convert_raw_dlc_to_keypoints(df: pd.DataFrame) -> pl.DataFrame:
    """Convert keypoints from old to new format."""
    # Impute na values with 0
    df = df.fillna(0)
    # Drop scorer level (always single value, useless)
    df.columns = df.columns.droplevel("scorer")
    # Name index as "frame"
    df.index.name = FRAME
    # Stack bodyparts + individuals + coords into rows, then unstack coords
    # to get x, y, likelihood as columns
    df = (
        df.stack(["individuals", "bodyparts", "coords"])  # noqa: PD010, PD013
        .unstack("coords")
        .reset_index()
    )
    # Convert to Polars long form
    return pl.from_pandas(df.reset_index()).select(
        pl.col(FRAME).cast(pl.Int64),
        pl.col("individuals").alias(INDIVIDUAL),
        pl.col("bodyparts").alias(BODYPART),
        pl.col(X).cast(pl.Float64),
        pl.col(Y).cast(pl.Float64),
        pl.col(LIKELIHOOD).cast(pl.Float64),
    )


def convert_keypoints_old_to_new_io(
    src_dir: Path, dst_dir: Path
) -> dict[str, pl.DataFrame]:
    """Convert keypoints from old to new format."""
    res = {}
    for _fp in src_dir.iterdir():
        name = _fp.stem
        if not (src_dir / f"{name}.parquet").exists():
            continue
        # Read h5 as pandas (DLC outputs pandas MultiIndex columns)
        df = pd.read_parquet(src_dir / f"{name}.parquet")
        # Convert
        df = convert_raw_dlc_to_keypoints(df)
        # Write to file
        write_df(df, dst_dir / f"{name}.parquet", KEYPOINTS_SCHEMA)
        # Save
        res[name] = df
    # Return
    return res
