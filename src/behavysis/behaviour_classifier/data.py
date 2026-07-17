"""Data loading and splitting for behavioural classifier."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

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
    PRED,
    PROB,
    TRUE_NEG,
    UNSURE,
    Array1D,
    Array2D,
)

if TYPE_CHECKING:
    from pathlib import Path

    from imblearn.under_sampling.base import BaseUnderSampler

# ── loading ─────────────────────────────────────────────────────


def load_feature_names(x_dir: Path) -> list[str]:
    """Load feature column names from the first features parquet file.

    Returns column names excluding "frame".
    """
    fp_ls = sorted(x_dir.iterdir())
    if not fp_ls:
        return []
    return [c for c in pl.read_parquet(fp_ls[0]).columns if c != FRAME]


def load_all_data(
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
            .select(
                FRAME,
                pl.col(ACTUAL).replace([FALSE_POS, UNSURE], [TRUE_NEG, TRUE_NEG]),
            )
        )

        aligned = x_df.join(y_df, on=FRAME, how="inner")
        if aligned.height == 0:
            continue

        pieces.append(aligned.with_columns(pl.lit(name).alias(EXPERIMENT)))

    return pl.concat(pieces, how="diagonal_relaxed")


# ── bout-related splitting ─────────────────────────────────────────────────────


def label_bouts(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``bout_id`` — integer label per contiguous (experiment, actual) run.

    TODO: move to behavysis.transforms.behaviours -> works better there.
    """
    return df.sort([EXPERIMENT, FRAME]).with_columns(
        (pl.col(ACTUAL) != pl.col(ACTUAL).shift(1))
        .or_(pl.col(EXPERIMENT) != pl.col(EXPERIMENT).shift(1))
        .cast(pl.Int64)
        .cum_sum()
        .backward_fill()
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
    idx = np.arange(len(df))
    y = df[ACTUAL].to_numpy()
    groups = df[group_name].to_numpy()

    n_splits = max(2, int(1 / test_size))
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    train_idx, test_idx = next(sgkf.split(idx, y, groups))
    return train_idx, test_idx


def agg_eval_df_by_bouts(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-frame eval data to per-bout rows.

    Each row represents one contiguous run (bout) of ACTUAL labels.  The
    aggregation preserves:

    * ``ACTUAL.max()`` — whether the bout is a real behavioural episode (1)
        or a non-behaviour gap (0).  This is the ground-truth label at bout
        level.
    * ``PROB.max()`` / ``PROB.mean()`` — peak and average model confidence
        across the bout.  Useful for ROC/PR curves (max) and for gauging
        how consistently the model recognises the behaviour (mean).
    * ``PRED.max()`` — whether *any* frame in the bout was predicted
        positive (standard SimBA bout-level match).
    * ``PRED.mean()`` — fraction of bout frames predicted positive.
        Answers: "how much of this bout did the model actually cover?"
        A bout with ``PRED.mean() < 0.5`` means the model missed more than
        half its frames, even if ``PRED.max() == 1``.  This enables
        IoU-style evaluation with custom coverage thresholds.
    * ``bout_n_frames`` — bout duration in frames.
    """
    return (
        label_bouts(df.sort([pl.col(EXPERIMENT), pl.col(FRAME)]))
        .group_by(BOUT_ID)
        .agg(
            pl.col(ACTUAL).max(),
            pl.col(PROB).max().alias(f"{PROB}_max"),
            pl.col(PROB).mean().alias(f"{PROB}_mean"),
            pl.col(PRED).max().alias(f"{PRED}_max"),
            pl.col(PRED).mean().alias(f"{PRED}_mean"),
            pl.len().alias("bout_n_frames"),
        )
        .sort(BOUT_ID)
    )


# ── X and y df preparing ─────────────────────────────────────────────────────


def df_get_features(df: pl.DataFrame) -> Array2D:
    """Given a df, only return features (filters out metadata and label columns)."""
    return (
        df.drop([EXPERIMENT, FRAME, ACTUAL, BOUT_ID], strict=False)
        .to_numpy()
        .astype(np.float32)
    )


def df_get_labels(df: pl.DataFrame) -> Array1D:
    """Given a df, return only the labels."""
    return df[ACTUAL].to_numpy()


# ── preprocessing ─────────────────────────────────────────────────────────────


def df_resample(df: pl.DataFrame, resampler: BaseUnderSampler) -> pl.DataFrame:
    """Resample."""
    # Make idx (has to be in shape (n,1))
    idx = np.arange(len(df)).reshape(-1, 1)
    # Sample and get sampled IDs
    sub_idx, _ = resampler.fit_resample(idx, df_get_labels(df))
    # Convert idx back to shape (n)
    sub_idx = sub_idx.reshape(-1)
    # Get sampled df
    return df[sub_idx]


# ── y prob smoothing ──────────────────────────────────────────────────────────


def smooth_prob(
    y_df: pl.DataFrame,
    smoothing_frames: int,
    agg_func: Literal["mean", "median"],
) -> pl.DataFrame:
    """Smoothing "prob" per-experiment.

    Assumes y_df is sorted with contiguous frames
    (or contiguous frames within each "experiment").
    Smoothing frames is either side of current.
    """
    # If no smoothing
    if smoothing_frames <= 0:
        return y_df
    # Get window size
    window_size = 2 * smoothing_frames + 1
    # Make smoothing agg expression
    expr = pl.col(PROB)
    if agg_func == "mean":
        expr = expr.rolling_mean(
            window_size=window_size,
            center=True,
            min_samples=1,
        )
    elif agg_func == "median":
        expr = expr.rolling_median(
            window_size=window_size,
            center=True,
            min_samples=1,
        )
    else:
        msg = f"Unsupported aggregation: {agg_func}"
        raise ValueError(msg)
    # If multiple experiments in df, then group by them
    if EXPERIMENT in y_df.columns:
        expr = expr.over(EXPERIMENT)
    # Compute and return
    return y_df.with_columns(expr)
