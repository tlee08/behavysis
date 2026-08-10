"""Behaviour utility functions operating on Polars DataFrames.

Predicted (long):     (frame, behaviour, prob, pred)
Scored (fully wide):   (frame, <behaviour1>, <sub1a>, <behaviour2>, ...)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl

from behavysis.constants import (
    BEHAVIOUR,
    BOUT_ID,
    DUR,
    EXPERIMENT,
    FRAME,
    PRED,
    PROB,
    START,
    STOP,
    TRUE_NEG,
    TRUE_POS,
    UNSURE,
)
from behavysis.models import Bout, Bouts, BoutStruct, ClassifierRef
from behavysis.schemas import make_scored_schema

COUNT = "count"


def get_group_cols(df: pl.DataFrame) -> list[str]:
    """Get the columns to group behaviours by."""
    group_cols = []
    if EXPERIMENT in df.columns:
        group_cols.append(EXPERIMENT)
    if BEHAVIOUR in df.columns:
        group_cols.append(BEHAVIOUR)
    return group_cols


def label_bouts(df: pl.DataFrame, label_col: str) -> pl.DataFrame:
    """Label contiguous behavioural bouts with globally unique IDs."""
    group_cols = get_group_cols(df)
    return df.sort([*group_cols, FRAME]).with_columns(
        pl.struct(*group_cols, label_col).rle_id().alias(BOUT_ID)
    )


def smooth_prob(
    df: pl.DataFrame,
    *,
    smoothing_frames: int = 1,
    agg_func: Literal["mean", "median"] = "median",
) -> pl.DataFrame:
    """Smoothing "prob" per-experiment.

    Assumes y_df is sorted with contiguous frames
    (or contiguous frames within each "experiment").
    Smoothing frames is either side of current.
    """
    if smoothing_frames <= 0:
        return df
    window_size = 2 * smoothing_frames + 1
    expr = pl.col(PROB)
    if agg_func == "mean":
        expr = expr.rolling_mean(window_size=window_size, center=True, min_samples=1)
    elif agg_func == "median":
        expr = expr.rolling_median(window_size=window_size, center=True, min_samples=1)
    else:
        msg = f"Unsupported aggregation: {agg_func}"
        raise ValueError(msg)
    group_cols = get_group_cols(df)
    expr = expr.over(group_cols)
    return df.sort([*group_cols, FRAME]).with_columns(expr)


def smooth_pred_bout(
    df: pl.DataFrame,
    *,
    min_gap: int = 3,
    min_bout: int = 3,
) -> pl.DataFrame:
    """Close short gaps, then remove short positive bouts."""
    df = label_bouts(df, PRED).with_columns(
        pl.when((pl.col(PRED) == TRUE_NEG) & (pl.len().over(BOUT_ID) <= min_gap))
        .then(TRUE_POS)
        .otherwise(pl.col(PRED))
        .alias(PRED)
    )
    df = label_bouts(df, PRED).with_columns(
        pl.when((pl.col(PRED) == TRUE_POS) & (pl.len().over(BOUT_ID) <= min_bout))
        .then(TRUE_NEG)
        .otherwise(pl.col(PRED))
        .alias(PRED)
    )
    return df.drop(BOUT_ID)


def vect2bouts(vect: pl.Series, offset: int = 0) -> pl.DataFrame:
    """Convert boolean vector to bouts DataFrame with start, stop, dur columns."""
    if vect.is_empty():
        return pl.DataFrame(schema={START: pl.Int64, STOP: pl.Int64, DUR: pl.Int64})
    z = np.concatenate(([TRUE_NEG], vect.to_numpy(), [TRUE_NEG]))
    starts = np.flatnonzero(~z[:-1] & z[1:])
    stops = np.flatnonzero(z[:-1] & ~z[1:]) - 1
    return pl.DataFrame(
        {
            START: pl.Series(starts + offset, dtype=pl.Int64),
            STOP: pl.Series(stops + offset, dtype=pl.Int64),
            DUR: pl.Series(stops - starts + 1, dtype=pl.Int64),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Predicted → scored (long → fully wide)
# ═══════════════════════════════════════════════════════════════════════════════


def predicted_to_scored(
    df: pl.DataFrame,
    classify_behaviour: dict[str, ClassifierRef],
) -> pl.DataFrame:
    """Convert predicted (long) to scored (fully wide) DataFrame.

    For each behaviour in ``classify_behaviour``, pivot the ``PRED``
    column into a behaviour-named column:
    - ``PRED == TRUE_POS`` → ``UNSURE`` (predicted bout, not yet scored)
    - otherwise → copy the PRED value (usually TRUE_NEG or FALSE_POS)

    Sub-behaviour columns are initialised to ``TRUE_NEG``.
    """
    frames = df.select(FRAME).unique().sort(FRAME)
    result_df = pl.DataFrame({FRAME: frames.get_column(FRAME)})

    for behaviour, ref in classify_behaviour.items():
        b_df = df.filter(pl.col(BEHAVIOUR) == behaviour).select(
            FRAME,
            pl.when(pl.col(PRED) == TRUE_POS)
            .then(pl.lit(UNSURE))
            .otherwise(pl.col(PRED))
            .alias(behaviour),
        )
        result_df = result_df.join(b_df, on=FRAME, how="left")
        result_df = result_df.with_columns(pl.col(behaviour).fill_null(TRUE_NEG))

        for sub in ref.sub_behaviour:
            result_df = result_df.with_columns(pl.lit(TRUE_NEG).alias(sub))

    expected_schema = make_scored_schema(classify_behaviour)
    return result_df.select(list(expected_schema.keys()))


# ═══════════════════════════════════════════════════════════════════════════════
# Scored ↔ Bouts (fully wide ↔ bout model)
# ═══════════════════════════════════════════════════════════════════════════════


def frames2bouts(
    df: pl.DataFrame,
    classify_behaviour: dict[str, ClassifierRef],
) -> Bouts:
    """Convert fully-wide scored DataFrame to Bouts model.

    For each behaviour column, reads values directly.
    Bouts are detected from any value in ``{TRUE_POS, UNSURE}``.
    """
    start_frame = df.select(FRAME).min().item()
    stop_frame = df.select(FRAME).max().item() + 1

    bouts_struct = _classify_to_bouts_struct(classify_behaviour)
    bouts_ls: list[Bout] = []

    for behaviour, ref in classify_behaviour.items():
        values = df.sort(FRAME).get_column(behaviour).to_numpy()
        bout_mask = np.isin(values, [TRUE_POS, UNSURE])

        if not bout_mask.any():
            continue

        bouts_df = vect2bouts(pl.Series(bout_mask), offset=start_frame)

        for row in bouts_df.iter_rows(named=True):
            bout_start = row[START]
            bout_stop = row[STOP]
            dur_val = row[DUR]

            bout_slice = df.filter(pl.col(FRAME).is_between(bout_start, bout_stop))
            actual_vals = bout_slice.get_column(behaviour)
            actual_mode = int(
                actual_vals.value_counts().sort(COUNT, descending=True).row(0)[0]
            )

            sub_behaviour = {}
            for sub in ref.sub_behaviour:
                if sub in df.columns:
                    vals = bout_slice.get_column(sub).drop_nulls()
                    if len(vals) > 0:
                        sub_behaviour[sub] = int(
                            vals.value_counts().sort(COUNT, descending=True).row(0)[0]
                        )

            bouts_ls.append(
                Bout(
                    start=bout_start,
                    stop=bout_stop,
                    dur=dur_val,
                    behaviour=behaviour,
                    actual=actual_mode,
                    sub_behaviour=sub_behaviour,
                )
            )

    return Bouts(
        start=start_frame,
        stop=stop_frame,
        bouts=bouts_ls,
        bout_struct=bouts_struct,
    )


def bouts2frames(bouts: Bouts) -> pl.DataFrame:
    """Convert Bouts model to fully-wide scored DataFrame.

    One row per frame. All behaviour/sub_behaviour columns initialised to
    ``TRUE_NEG``, then filled in from bout data.
    """
    frames = np.arange(bouts.start, bouts.stop, dtype=np.int64)
    classify_behaviour = _bouts_struct_to_classify(bouts.bout_struct)
    schema = make_scored_schema(classify_behaviour)

    rows = [
        {FRAME: int(f)} | {col: TRUE_NEG for col in schema if col != FRAME}
        for f in frames
    ]
    df = pl.DataFrame(rows, schema=schema)

    for bout in bouts.bouts:
        mask = (pl.col(FRAME) >= bout.start) & (pl.col(FRAME) <= bout.stop)

        df = df.with_columns(
            pl.when(mask)
            .then(pl.lit(bout.actual))
            .otherwise(pl.col(bout.behaviour))
            .alias(bout.behaviour),
        )
        for k, v in bout.sub_behaviour.items():
            df = df.with_columns(
                pl.when(mask).then(pl.lit(v)).otherwise(pl.col(k)).alias(k),
            )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _classify_to_bouts_struct(
    classify_behaviour: dict[str, ClassifierRef],
) -> list[BoutStruct]:
    """Convert classify_behaviour dict to list of BoutStruct."""
    return [
        BoutStruct(behaviour=behaviour, sub_behaviour=ref.sub_behaviour)
        for behaviour, ref in classify_behaviour.items()
    ]


def _bouts_struct_to_classify(
    bouts_struct: list[BoutStruct],
) -> dict[str, ClassifierRef]:
    """Convert list of BoutStruct back to classify_behaviour dict."""
    return {
        bs.behaviour: ClassifierRef(sub_behaviour=bs.sub_behaviour)
        for bs in bouts_struct
    }
