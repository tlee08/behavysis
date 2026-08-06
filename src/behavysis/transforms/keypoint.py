"""Keypoints utility functions operating on Polars long-form DataFrames.

All functions operate on DataFrames conforming to
``KEYPOINTS_SCHEMA``: (frame, individual, bodypart, x, y, likelihood).
"""

import polars as pl

from behavysis.constants import BODYPART, FRAME, INDIVIDUAL, PROCESSED, SINGLE, X, Y


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


def bodypart_avg_xy(
    df: pl.DataFrame,
    indiv: str,
    bpts: list[str],
) -> pl.DataFrame:
    """Average x and y coordinates across bodyparts per frame for an individual."""
    return (
        df.filter(pl.col(INDIVIDUAL) == indiv, pl.col(BODYPART).is_in(bpts))
        .group_by(FRAME)
        .agg([pl.col(X).mean().alias(X), pl.col(Y).mean().alias(Y)])
        .sort(FRAME)
    )
