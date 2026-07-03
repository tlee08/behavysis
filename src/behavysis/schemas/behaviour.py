"""Behaviour utility functions operating on Polars DataFrames.

Predicted: (frame, behaviour, prob, pred)
Scored: (frame, behaviour, pred, actual, [user_defined...])
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from behavysis.constants import (
    ACTUAL,
    BEHAVIOUR,
    DUR,
    FRAME,
    PRED,
    START,
    STOP,
    TRUE_NEG,
    TRUE_POS,
    UNSURE,
)
from behavysis.models import Bout, Bouts, BoutStruct

from .schemas import BEHAVIOUR_SCORED_BASE

COUNT = "count"


def vect2bouts(vect: pl.Series, offset: int = 0) -> pl.DataFrame:
    """Convert boolean vector to bouts DataFrame with start, stop, dur columns.

    Parameters
    ----------
    vect : pl.Series
        Boolean series where True indicates a bout frame.
    offset : int
        Frame offset to add to start/stop values.

    Returns:
    -------
    pl.DataFrame
        DataFrame with ``start``, ``stop``, ``dur`` columns.
    """
    if vect.is_empty():
        return pl.DataFrame(schema={START: pl.Int64, STOP: pl.Int64, DUR: pl.Int64})
    # Use numpy for edge detection (fast and simple)
    z = np.concatenate(([TRUE_NEG], vect.to_numpy(), [TRUE_NEG]))
    starts = np.flatnonzero(~z[:-1] & z[1:])
    stops = np.flatnonzero(z[:-1] & ~z[1:]) - 1
    # Return df where each row is each bout's start-stop-duration
    return pl.DataFrame(
        {
            START: pl.Series(starts + offset, dtype=pl.Int64),
            STOP: pl.Series(stops + offset, dtype=pl.Int64),
            DUR: pl.Series(stops - starts + 1, dtype=pl.Int64),
        },
    )


def merge_bouts(vect: pl.Series, min_window_frames: int) -> pl.Series:
    """Merge behaviour bouts that are close together.

    If the gap between two bouts is less than ``min_window_frames``, merge them.
    """
    if vect.is_empty():
        return vect
    # Find gaps (frames that are not TRUE_POS)
    nonbouts_df = vect2bouts(vect != TRUE_POS)
    # To numpy for run-length encoding operations
    arr = vect.to_numpy().copy()
    # For each non-bout, if less than min duration, then fill
    for nonbout_row in nonbouts_df.iter_rows(named=True):
        if nonbout_row[DUR] < min_window_frames:
            arr[nonbout_row[START] : nonbout_row[STOP] + 1] = TRUE_POS

    return pl.Series(arr)


def predicted2scored(
    df: pl.DataFrame,
    bouts_struct: list[BoutStruct],
) -> pl.DataFrame:
    """Convert predicted behaviour DataFrame to scored behaviour DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Predicted behaviour DataFrame (BEHAVIOUR_PREDICTED_SCHEMA).
    bouts_struct : list[BoutStruct]
        Bout structure definitions from config.

    Returns:
    -------
    pl.DataFrame
        Scored behaviour DataFrame with pred, actual, and user_defined columns.
    """
    # Start with frame, behaviour, pred
    result_df = df.select([FRAME, BEHAVIOUR, PRED])
    # actual = pred but positive predictions become UNSURE (unscored)
    result_df = result_df.with_columns(
        pl.when(pl.col(PRED) == TRUE_POS)
        .then(pl.lit(UNSURE))
        .otherwise(pl.col(PRED))
        .alias(ACTUAL),
    )
    # Drop the pred column
    result_df = result_df.drop(PRED)
    # Add user_defined columns initialised to TRUE_NEG
    for bout_struct in bouts_struct:
        for user_col in bout_struct.sub_behaviour:
            result_df = result_df.with_columns(pl.lit(TRUE_NEG).alias(user_col))
    # Return result df
    return result_df


def get_bouts_struct(df: pl.DataFrame) -> list[BoutStruct]:
    """Extract BoutStruct from DataFrame columns.

    Parameters
    ----------
    df : pl.DataFrame
        Scored behaviour DataFrame.

    Returns:
    -------
    list[BoutStruct]
        Bout structure definitions.
    """
    # Get user_defined columns (everything except frame, behaviour, actual)
    base_cols = {FRAME, BEHAVIOUR, ACTUAL}
    user_cols = [c for c in df.columns if c not in base_cols]

    behaviours_ls = df.select(BEHAVIOUR).unique().sort(BEHAVIOUR).to_series().to_list()

    bouts_struct = []
    for behaviour in behaviours_ls:
        # Determine which user_defined columns apply to this behaviour
        # (those that have non-null values for this behaviour)
        sub_behaviour_ls = []
        for col in user_cols:
            null_count = (
                df.filter(pl.col(BEHAVIOUR) == behaviour)
                .select(
                    pl.col(col).null_count(),
                )
                .item()
            )
            total = df.filter(pl.col(BEHAVIOUR) == behaviour).height
            if total > 0 and null_count < total:
                sub_behaviour_ls.append(col)
        bouts_struct.append(
            BoutStruct(behaviour=behaviour, sub_behaviour=sub_behaviour_ls),
        )

    return bouts_struct


def frames2bouts(df: pl.DataFrame) -> Bouts:
    """Convert frame-level scored DataFrame to Bouts model.

    Parameters
    ----------
    df : pl.DataFrame
        Scored behaviour DataFrame.

    Returns:
    -------
    Bouts
        Bouts model with start, stop, and list of Bout objects.
    """
    start_frame = df.select(FRAME).min().item()
    stop_frame = df.select(FRAME).max().item() + 1
    behaviours_ls = df.select(BEHAVIOUR).unique().sort(BEHAVIOUR).to_series().to_list()

    bouts_struct = get_bouts_struct(df)
    bouts_ls = []

    for behaviour in behaviours_ls:
        behaviour_df = df.filter(pl.col(BEHAVIOUR) == behaviour).sort(FRAME)

        # Get boolean pred series for this behaviour (sorted by frame)
        pred_bool = behaviour_df.select(PRED).to_series() == TRUE_POS

        if pred_bool.sum() == 0:
            continue

        # Compute frame offset for this behaviour's frame range
        frame_offset = behaviour_df.select(FRAME).min().item()
        bouts_df = vect2bouts(pred_bool, offset=frame_offset)

        for row in bouts_df.iter_rows(named=True):
            bout_start = row[START]
            bout_stop = row[STOP]
            dur_val = row[DUR]

            # Get actual value (mode of actual column within bout)
            bout_slice = behaviour_df.filter(
                pl.col(FRAME).is_between(bout_start, bout_stop),
            )
            actual_vals = bout_slice.select(ACTUAL).to_series()
            actual_mode = int(
                actual_vals.value_counts().sort(COUNT, descending=True).row(0)[0],
            )

            # Get user_defined values
            user_defined = {}
            for col in [
                c for c in df.columns if c not in {FRAME, BEHAVIOUR, PRED, ACTUAL}
            ]:
                if col in bout_slice.columns:
                    vals = bout_slice.select(col).to_series().drop_nulls()
                    if len(vals) > 0:
                        user_defined[col] = int(
                            vals.value_counts().sort(COUNT, descending=True).row(0)[0],
                        )

            bouts_ls.append(
                Bout(
                    start=bout_start,
                    stop=bout_stop,
                    dur=dur_val,
                    behaviour=behaviour,
                    actual=actual_mode,
                    sub_behaviour=user_defined,
                ),
            )

    return Bouts(
        start=start_frame,
        stop=stop_frame,
        bouts=bouts_ls,
        bout_struct=bouts_struct,
    )


def bouts2frames(bouts: Bouts) -> pl.DataFrame:
    """Convert Bouts model to frame-level scored DataFrame.

    Parameters
    ----------
    bouts : Bouts
        Bouts model with start, stop, and list of Bout objects.

    Returns:
    -------
    pl.DataFrame
        Long-form scored behaviour DataFrame.
    """
    behaviours = [b.behaviour for b in bouts.bout_struct]
    user_cols = list({col for b in bouts.bout_struct for col in b.sub_behaviour})

    # Build frame x behaviour grid
    frames = np.arange(bouts.start, bouts.stop, dtype=np.int64)
    rows = []

    for behaviour in behaviours:
        for f in frames:
            row = {
                FRAME: int(f),
                BEHAVIOUR: behaviour,
                PRED: TRUE_NEG,
                ACTUAL: TRUE_NEG,
            }
            for col in user_cols:
                row[col] = TRUE_NEG
            rows.append(row)

    df = pl.DataFrame(
        rows,
        schema={**BEHAVIOUR_SCORED_BASE, **dict.fromkeys(user_cols, pl.Int64)},
    )

    # Fill in bout values
    for bout in bouts.bouts:
        mask = (
            (pl.col(FRAME) >= bout.start)
            & (pl.col(FRAME) <= bout.stop)
            & (pl.col(BEHAVIOUR) == bout.behaviour)
        )
        df = df.with_columns(
            pl.when(mask).then(pl.lit(TRUE_POS)).otherwise(pl.col(PRED)).alias(PRED),
            pl.when(mask)
            .then(pl.lit(bout.actual))
            .otherwise(pl.col(ACTUAL))
            .alias(ACTUAL),
        )
        for k, v in bout.sub_behaviour.items():
            df = df.with_columns(
                pl.when(mask).then(pl.lit(v)).otherwise(pl.col(k)).alias(k),
            )

    return df


def import_boris_tsv(
    fp: Path,
    behaviour_ls: list[str],
    start_frame: int,
    stop_frame: int,
) -> pl.DataFrame:
    """Import BORIS TSV file to scored behaviour DataFrame.

    Parameters
    ----------
    fp : Path
        Path to BORIS TSV file.
    behaviour_ls : list[str]
        List of behaviour names to import.
    start_frame : int
        First frame of the experiment.
    stop_frame : int
        Last frame of the experiment (exclusive).

    Returns:
    -------
    pl.DataFrame
        Long-form scored behaviour DataFrame.
    """
    df_boris = pd.read_csv(fp, sep="\t")
    boris_behaviours = df_boris[BEHAVIOUR].unique()

    if not np.isin(behaviour_ls, boris_behaviours).all():
        msg = (
            f"Some behaviours not in BORIS file.\n"
            f"Requested: {behaviour_ls}\n"
            f"BORIS: {boris_behaviours}"
        )
        raise ValueError(msg)

    frames = np.arange(start_frame, stop_frame, dtype=np.int64)

    # Build frame x behaviour grid
    rows = [
        {
            FRAME: int(f),
            BEHAVIOUR: behaviour,
            PRED: TRUE_NEG,
            ACTUAL: TRUE_NEG,
        }
        for f in frames
        for behaviour in behaviour_ls
    ]
    df = pl.DataFrame(rows, schema=BEHAVIOUR_SCORED_BASE)

    # Apply BORIS events
    for _, row_boris in df_boris.iterrows():
        behaviour = row_boris[BEHAVIOUR]
        frame = row_boris["Image index"]
        status = row_boris["Behaviour type"]

        if behaviour not in behaviour_ls:
            continue
        val = TRUE_POS if status == START else TRUE_NEG

        mask = (pl.col(FRAME) >= frame) & (pl.col(BEHAVIOUR) == behaviour)
        df = df.with_columns(
            pl.when(mask).then(pl.lit(val)).otherwise(pl.col(PRED)).alias(PRED),
            pl.when(mask).then(pl.lit(val)).otherwise(pl.col(ACTUAL)).alias(ACTUAL),
        )

    return df
