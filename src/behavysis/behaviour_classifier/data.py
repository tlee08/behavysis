"""Data loading and splitting for behavioural classifier."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold

from behavysis.constants import (
    BEHAVIOUR,
    BOUT_ID,
    EXPERIMENT,
    FALSE_POS,
    FRAME,
    PRED,
    PROB,
    TRUE_NEG,
    UNSURE,
    Array1D,
)
from behavysis.transforms import label_bouts

if TYPE_CHECKING:
    from imblearn.under_sampling.base import BaseUnderSampler

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
            pl.col(behaviour_name)
            .replace([FALSE_POS, UNSURE], [TRUE_NEG, TRUE_NEG])
            .alias(ACTUAL),
        )
        aligned = x_df.join(y_df, on=FRAME, how="inner")
        if aligned.height == 0:
            continue
        pieces.append(aligned.with_columns(pl.lit(name).alias(EXPERIMENT)))

    return pl.concat(pieces, how="diagonal_relaxed")


# ── X and y extracting ───────────────────────────────────────────────


def df_get_features(df: pl.DataFrame, *, label_col: str) -> pl.DataFrame:
    """Given a df, return only features (drops metadata and label columns)."""
    return df.drop(
        [EXPERIMENT, FRAME, BEHAVIOUR, BOUT_ID, label_col], strict=False
    ).cast(pl.Float32)


def df_get_labels(df: pl.DataFrame, *, label_col: str) -> pl.Series:
    """Given a df, return only the labels."""
    return df[label_col]


# ── splitting ────────────────────────────────────────────────────────


def stratified_split_by_group(
    df: pl.DataFrame,
    test_size: float,
    group_name: str,
    random_state: int = 42,
    *,
    label_col: str,
) -> tuple[Array1D, Array1D]:
    """Split into train/test, grouping contiguous label runs together."""
    idx = np.arange(len(df))
    y = df[label_col].to_numpy()
    groups = df[group_name].to_numpy()

    n_splits = max(2, int(1 / test_size))
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    train_idx, test_idx = next(sgkf.split(idx, y, groups))
    return train_idx, test_idx


# ── bout aggregation ─────────────────────────────────────────────────


def agg_eval_df_by_bouts(df: pl.DataFrame, *, label_col: str) -> pl.DataFrame:
    """Aggregate per-frame eval data to per-bout rows."""
    return (
        label_bouts(df, label_col)
        .group_by(BOUT_ID)
        .agg(
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


def df_resample(
    df: pl.DataFrame, resampler: BaseUnderSampler, *, label_col: str
) -> pl.DataFrame:
    """Resample."""
    idx = np.arange(len(df)).reshape(-1, 1)
    sub_idx, _ = resampler.fit_resample(idx, df_get_labels(df, label_col=label_col))
    sub_idx = sub_idx.reshape(-1)
    return df[sub_idx]
