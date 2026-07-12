"""Data loading and splitting for behavioural classifier."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold

from behavysis.constants import (
    ACTUAL,
    BEHAVIOUR,
    BOUT_ID,
    EXPERIMENT,
    FALSE_POS,
    FRAME,
    UNSURE,
    Array1D,
)

if TYPE_CHECKING:
    from pathlib import Path


def load_feature_names(x_dir: Path) -> list[str]:
    """Load feature column names from the first features parquet file.

    Returns column names excluding "frame".
    """
    fp_ls = sorted(x_dir.iterdir())
    if not fp_ls:
        return []
    return [c for c in pl.read_parquet(fp_ls[0]).columns if c != FRAME]


def load_training_data(
    x_dir: Path,
    y_dir: Path,
    behaviour_name: str,
) -> pl.DataFrame:
    """Load features and scored labels, aligned by frame per experiment.

    Each experiment's features parquet is inner-joined with its scored
    labels parquet on ``frame``, guaranteeing row-for-row alignment.
    The result is a single DataFrame with columns:

    - ``experiment`` — experiment name (from parquet filename stem)
    - ``frame`` — frame number
    - ``actual`` — label (unsure replaced with false positive)
    - one column per feature (Float64)

    Parameters
    ----------
    x_dir : Path
        Directory of feature parquet files (``5_features_extracted/``).
    y_dir : Path
        Directory of scored behaviour parquet files (``7_behaviour_scored/``).
    behaviour_name : str
        Target behaviour to extract as the ``actual`` column.

    Returns:
    -------
    pl.DataFrame
        Aligned training data with metadata + feature columns.
    """
    x_fps = {fp.stem: fp for fp in sorted(x_dir.iterdir())}
    y_fps = {fp.stem: fp for fp in sorted(y_dir.iterdir())}
    common = sorted(set(x_fps) & set(y_fps))

    pieces: list[pl.DataFrame] = []
    for name in common:
        x_df = pl.read_parquet(x_fps[name])

        y_df = (
            pl.read_parquet(y_fps[name])
            .filter(pl.col(BEHAVIOUR) == behaviour_name)
            .select(FRAME, pl.col(ACTUAL).replace({UNSURE: FALSE_POS}).alias(ACTUAL))
        )

        aligned = x_df.join(y_df, on=FRAME, how="inner")
        if aligned.height == 0:
            continue

        pieces.append(aligned.with_columns(pl.lit(name).alias(EXPERIMENT)))

    return pl.concat(pieces, how="diagonal_relaxed")


def label_bouts(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``bout_id`` — integer label per contiguous (experiment, actual) run."""
    return df.with_columns(
        (pl.col(ACTUAL) != pl.col(ACTUAL).shift(1))
        .or_(pl.col(EXPERIMENT) != pl.col(EXPERIMENT).shift(1))
        .cast(pl.Int64)
        .cum_sum()
        .alias(BOUT_ID)
    )


def stratified_split_by_group(
    df: pl.DataFrame,
    test_size: float,
    group_name: str,
    random_state: int = 42,
) -> tuple[Array1D, Array1D]:
    """Split into train/test, grouping contiguous label runs together.

    Group name like: "bout_id", "experiment"
    """
    df = label_bouts(df)
    x = df.drop([EXPERIMENT, FRAME, ACTUAL, BOUT_ID]).to_numpy()
    y = df[ACTUAL].to_numpy()
    groups = df[group_name].to_numpy()

    n_splits = max(2, int(1 / test_size))
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    train_idx, test_idx = next(sgkf.split(x, y, groups))

    train_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    return train_mask, test_mask
