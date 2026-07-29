"""Behaviour utility functions operating on Polars DataFrames.

Predicted: (frame, behaviour, prob, pred)
Scored: (frame, behaviour, pred, actual, [sub_behaviour...])
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
from loguru import logger

from behavysis.constants import (
    ACTUAL,
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
from behavysis.models import (
    Bout,
    Bouts,
    BoutStruct,
    ExperimentMetadata,
)
from behavysis.schemas import BEHAVIOUR_SCORED_BASE, write_df
from behavysis.utils import log_file_exists

if TYPE_CHECKING:
    from pathlib import Path


COUNT = "count"


def get_group_cols(df: pl.DataFrame) -> list[str]:
    """Get the columns to group behaviours by."""
    # TODO: change when we add sub_behaviour column
    group_cols = []
    if EXPERIMENT in df.columns:
        group_cols.append(EXPERIMENT)
    group_cols.append(BEHAVIOUR)
    return group_cols


def label_bouts(df: pl.DataFrame, label_col: str) -> pl.DataFrame:
    """Label contiguous behavioural bouts with globally unique IDs."""
    # If multiple experiments in df, then group by them
    group_cols = get_group_cols(df)
    # Return
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
    # If no smoothing
    if smoothing_frames <= 0:
        return df
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
    group_cols = get_group_cols(df)
    expr = expr.over(group_cols)
    # Sort, compute and return
    return df.sort([*group_cols, FRAME]).with_columns(expr)


def smooth_pred_bout(
    df: pl.DataFrame,
    *,
    min_gap: int = 3,
    min_bout: int = 3,
) -> pl.DataFrame:
    """Close short gaps, then remove short positive bouts."""
    # Label and merge short TRUE_NEG bouts
    df = label_bouts(df, PRED).with_columns(
        pl.when((pl.col(PRED) == TRUE_NEG) & (pl.len().over(BOUT_ID) <= min_gap))
        .then(TRUE_POS)
        .otherwise(pl.col(PRED))
        .alias(PRED)
    )
    # Label and drop short TRUE_POS bouts
    df = label_bouts(df, PRED).with_columns(
        pl.when((pl.col(PRED) == TRUE_POS) & (pl.len().over(BOUT_ID) <= min_bout))
        .then(TRUE_NEG)
        .otherwise(pl.col(PRED))
        .alias(PRED)
    )
    # Drop label and return
    return df.drop(BOUT_ID)


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


def predicted_to_scored(
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
        Scored behaviour DataFrame with pred, actual, and sub_behaviour columns.
    """
    # TODO: also return column schema for data validation later
    result_df = df.select([FRAME, BEHAVIOUR, PRED])
    result_df = result_df.with_columns(
        pl.when(pl.col(PRED) == TRUE_POS)
        .then(pl.lit(UNSURE))
        .otherwise(pl.col(PRED))
        .alias(ACTUAL),
    )
    result_df = result_df.drop(PRED)
    for bout_struct in bouts_struct:
        for user_col in bout_struct.sub_behaviour:
            result_df = result_df.with_columns(pl.lit(TRUE_NEG).alias(user_col))
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
    base_cols = {FRAME, BEHAVIOUR, ACTUAL}
    user_cols = [c for c in df.columns if c not in base_cols]

    behaviours_ls = df.select(BEHAVIOUR).unique().sort(BEHAVIOUR).to_series().to_list()

    bouts_struct = []
    for behaviour in behaviours_ls:
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
        pred_bool = behaviour_df.select(PRED).to_series() == TRUE_POS

        if pred_bool.sum() == 0:
            continue

        frame_offset = behaviour_df.select(FRAME).min().item()
        bouts_df = vect2bouts(pred_bool, offset=frame_offset)

        for row in bouts_df.iter_rows(named=True):
            bout_start = row[START]
            bout_stop = row[STOP]
            dur_val = row[DUR]

            bout_slice = behaviour_df.filter(
                pl.col(FRAME).is_between(bout_start, bout_stop),
            )
            actual_vals = bout_slice.select(ACTUAL).to_series()
            actual_mode = int(
                actual_vals.value_counts().sort(COUNT, descending=True).row(0)[0],
            )

            sub_behaviour = {}
            for col in [
                c for c in df.columns if c not in {FRAME, BEHAVIOUR, PRED, ACTUAL}
            ]:
                if col in bout_slice.columns:
                    vals = bout_slice.select(col).to_series().drop_nulls()
                    if len(vals) > 0:
                        sub_behaviour[col] = int(
                            vals.value_counts().sort(COUNT, descending=True).row(0)[0],
                        )

            bouts_ls.append(
                Bout(
                    start=bout_start,
                    stop=bout_stop,
                    dur=dur_val,
                    behaviour=behaviour,
                    actual=actual_mode,
                    sub_behaviour=sub_behaviour,
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


def import_boris_csv(
    fp: Path,
    behaviour_ls: list[str],
    start_frame: int,
    stop_frame: int,
    fps: int,
    *,
    point_window_sec: float = 0.0,
    pos_value: int = TRUE_POS,
) -> pl.DataFrame:
    """Import BORIS CSV to BEHAVIOUR_SCORED_BASE long-form DataFrame.

    Parameters
    ----------
    fp : Path
        Path to BORIS CSV file.
    behaviour_ls : list[str]
        List of behaviour names to import. Others in the CSV are skipped.
    start_frame : int
        First frame of the experiment.
    stop_frame : int
        Last frame of the experiment (exclusive).
    fps : int
        Frames per second for converting *point_window_sec* to frames.
    point_window_sec : float
        For POINT behaviours, mark ±window seconds around each event as
        ``actual=TRUE_POS``. Default 0.0 (single frame only).
    pos_value: int
        What to use for the "is_behaviour". Defaults to TRUE_POS = 1.

    Returns:
    --------
    pl.DataFrame
        Long-form DataFrame with schema ``BEHAVIOUR_SCORED_BASE``:
        ``{frame, behaviour, actual}``.
    """
    # Read boris df
    # Inferring frames from time so we are FPS agnostic
    df_boris = (
        pl.read_csv(fp)
        .rename({"Behavior": BEHAVIOUR})
        .with_columns(
            (pl.col("Time") * fps).round().cast(pl.Int64).alias(FRAME),
            pl.col("Behavior type").str.strip_chars().str.to_uppercase().alias("type"),
        )
    )
    # Check behaviour exists
    boris_behaviours = df_boris[BEHAVIOUR].unique().to_list()
    missing = [b for b in behaviour_ls if b not in boris_behaviours]
    if missing:
        logger.warning(
            "Behaviours not in BORIS file: {}\nBORIS: {}",
            missing,
            boris_behaviours,
        )
    # Make window
    window = int(point_window_sec * fps)
    frames = np.arange(start_frame, stop_frame, dtype=np.int64)
    # For each given behaviour, construct the fbf df from boris df
    fbf_df_ls: list[pl.DataFrame] = []
    for behaviour in behaviour_ls:
        _df = pl.DataFrame(
            {FRAME: frames, BEHAVIOUR: behaviour, ACTUAL: TRUE_NEG},
            schema=BEHAVIOUR_SCORED_BASE,
        )
        # Filter boris_df by behaviour and sort by frame
        evts_df = df_boris.filter(pl.col(BEHAVIOUR) == behaviour).sort(FRAME)
        for row in evts_df.iter_rows(named=True):
            f = int(row[FRAME])
            typ = row["type"]
            # If START or STOP, then set > curr_frame accordingly
            if typ in ("START", "STOP"):
                val = pos_value if typ == "START" else TRUE_NEG
                _df = _df.with_columns(
                    pl.when(pl.col(FRAME) >= f)
                    .then(val)
                    .otherwise(pl.col(ACTUAL))
                    .alias(ACTUAL),
                )
            # If POINT, then set nearby window to pos_value
            elif typ == "POINT":
                lo = max(f - window, start_frame)
                hi = min(f + window, stop_frame - 1)
                _df = _df.with_columns(
                    pl.when(pl.col(FRAME).is_between(lo, hi))
                    .then(pos_value)
                    .otherwise(pl.col(ACTUAL))
                    .alias(ACTUAL),
                )
        # Add to list
        fbf_df_ls.append(_df)
    # Concatenate fbf behaviours and return
    return pl.concat(fbf_df_ls)


def boris_to_behaviour(
    src_fp: Path,
    dst_fp: Path,
    metadata: ExperimentMetadata,
    behaviour_ls: list[str],
    *,
    overwrite: bool,
    point_window_sec: float = 0.0,
    fps: int = 50,
) -> None:
    """Import BORIS CSV and write scored behaviour parquet."""
    if not overwrite and dst_fp.exists():
        log_file_exists(dst_fp)
        return

    df = import_boris_csv(
        src_fp,
        behaviour_ls,
        metadata.require_start_frame(),
        metadata.require_stop_frame() + 1,
        point_window_sec=point_window_sec,
        fps=fps,
    )
    write_df(df, dst_fp, BEHAVIOUR_SCORED_BASE)
    logger.info("boris csv to behaviour")
