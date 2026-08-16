"""Data loading and splitting for behavioural classifier."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold

from behavysis.constants import (
    BEHAVIOUR,
    BOUT_ID,
    EXPERIMENT,
    FRAME,
    PRED,
    PROB,
    TRUE_NEG,
    TRUE_POS,
    Array1DInt,
)
from behavysis.transforms import label_bouts

if TYPE_CHECKING:
    from pathlib import Path

ACTUAL = "actual"


# ── loading ───────────────────────────────────────────────────────────


def load_all_data(
    x_dir: Path,
    y_dir: Path,
    behaviour_name: str,
) -> pl.DataFrame:
    """Load features and scored labels, aligned by frame per experiment.

    Renames the behaviour column to ``"actual"`` for internal classifier use.

    Returns a DataFrame with columns:
        EXPERIMENT, FRAME, actual, ...feature columns
    """
    x_fps = {fp.stem: fp for fp in sorted(x_dir.iterdir())}
    y_fps = {fp.stem: fp for fp in sorted(y_dir.iterdir())}
    common = sorted(set(x_fps) & set(y_fps))

    pieces: list[pl.DataFrame] = []
    for name in common:
        x_df = pl.read_parquet(x_fps[name])
        y_df = pl.read_parquet(y_fps[name]).select(
            FRAME,
            pl.when(pl.col(behaviour_name) == TRUE_POS)
            .then(TRUE_POS)
            .otherwise(TRUE_NEG)
            .alias(ACTUAL),
        )
        aligned = x_df.join(y_df, on=FRAME, how="inner")
        if aligned.height == 0:
            continue
        pieces.append(aligned.with_columns(pl.lit(name).alias(EXPERIMENT)))

    return pl.concat(pieces, how="diagonal_relaxed").sort([EXPERIMENT, FRAME])


# ── X and y extracting ───────────────────────────────────────────────


def df_get_features(df: pl.DataFrame, *, label_col: str = ACTUAL) -> pl.DataFrame:
    """Given a df, return only features (drops metadata and label columns)."""
    return df.drop(
        [EXPERIMENT, FRAME, BEHAVIOUR, BOUT_ID, label_col], strict=False
    ).cast(pl.Float32)


def df_get_labels(df: pl.DataFrame, *, label_col: str = ACTUAL) -> pl.Series:
    """Given a df, return only the labels."""
    return df.get_column(label_col)


# ── splitting ────────────────────────────────────────────────────────


def stratified_split_by_group(
    df: pl.DataFrame,
    test_size: float,
    group_name: str,
    random_state: int = 42,
    *,
    label_col: str = ACTUAL,
) -> tuple[Array1DInt, Array1DInt]:
    """Split into train/test, grouping contiguous label runs together."""
    idx = np.arange(len(df))
    y = df.get_column(label_col).to_numpy()
    groups = df.get_column(group_name).to_numpy()

    n_splits = max(2, int(1 / test_size))
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    train_idx, test_idx = next(sgkf.split(idx, y, groups))
    return train_idx, test_idx


# ── bout aggregation ─────────────────────────────────────────────────


def agg_eval_df_by_bouts(df: pl.DataFrame, *, label_col: str = ACTUAL) -> pl.DataFrame:
    """Aggregate per-frame eval data to per-bout rows."""
    return (
        label_bouts(df, label_col)
        .group_by(BOUT_ID)
        .agg(
            pl.col(EXPERIMENT).first(),
            pl.col(FRAME).first().alias("bout_start_frame"),
            pl.col(label_col).max(),
            pl.col(PROB).max(),
            pl.col(PROB).mean().alias(f"{PROB}_mean"),
            pl.col(PRED).max(),
            pl.col(PRED).mean().alias(f"{PRED}_mean"),
            pl.len().alias("bout_n_frames"),
        )
        .sort(BOUT_ID)
    )


# ── preprocessing ────────────────────────────────────────────────────


def df_stride_sample(
    df: pl.DataFrame,
    stride_frames: int,
) -> pl.DataFrame:
    """Bout-aware stride sampling.

    Keeps every ``stride_frames``-th frame *within each bout* (a contiguous
    run of the label inside an experiment), so every bout contributes at
    least one frame and short bouts are never dropped.  Assumes ``df`` is
    already bout-labelled (has a ``BOUT_ID`` column).

    Assumes ``df`` is sorted by ``EXPERIMENT`` and ``FRAME``.
    """
    if stride_frames <= 1:
        return df
    return (
        df.with_columns(pl.int_range(pl.len()).over([EXPERIMENT, BOUT_ID]).alias("_i"))
        .filter(pl.col("_i") % stride_frames == 0)
        .drop("_i")
    )


def df_under_sample_by_group(
    df: pl.DataFrame,
    strategy: float,
    *,
    label_col: str = ACTUAL,
    group_col: str = EXPERIMENT,
    seed: int = 42,
) -> pl.DataFrame:
    """Random under-sampling of the majority class, per group.

    Keeps every minority sample and, within each group, ``ceil(n_minority /
    strategy)`` majority samples chosen uniformly at random.  ``strategy``
    follows imblearn's ``sampling_strategy`` float convention (the desired
    ratio of the minority class over the majority class after sampling).
    Grouping guarantees no group (experiment) is under-represented after
    sampling.  Groups with no minority keep a single majority sample.

    Assumes ``df`` is sorted by ``group_col`` and ``FRAME``.
    """
    if strategy is None:
        return df
    rng = np.random.default_rng(seed)
    pieces: list[pl.DataFrame] = []
    for sub in df.partition_by([group_col], maintain_order=True):
        minority = sub.filter(pl.col(label_col) == 1)
        majority = sub.filter(pl.col(label_col) == 0)
        n_keep = int(np.ceil(minority.height / strategy)) if minority.height > 0 else 1
        n_keep = min(n_keep, majority.height)
        if n_keep < majority.height:
            majority = majority.gather(
                rng.choice(majority.height, size=n_keep, replace=False)
            )
        pieces.append(pl.concat([minority, majority]))
    return pl.concat(pieces)
