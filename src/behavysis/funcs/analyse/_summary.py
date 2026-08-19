"""Analysis result orchestration: binned summaries and AnalysisResult wrapping.

These functions consume analysis DataFrames, apply aggregate/binning transforms,
and wrap results in AnalysisResult objects with save semantics.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from behavysis.constants import (
    AGG,
    BIN_SEC,
    BINNED,
    CUSTOM,
    DF_IO_FORMAT,
    DUR,
    FRAME,
    GROUP,
    MEASURE,
    SUMMARY,
    VALUE,
)
from behavysis.schemas import BINNED_SCHEMA, SUMMARY_SCHEMA, write_df
from behavysis.transforms import vect2bouts

from ._helper import AnalysisResult


def agg_quantitative(df: pl.DataFrame, fps: float) -> pl.DataFrame:
    """Summarize quantitative data: mean, std, min, Q1, median, Q3, max, sum.

    Parameters
    ----------
    df : pl.DataFrame
        ANALYSIS_SCHEMA DataFrame.
    fps : float
        Frames per second (unused, kept for API compatibility).

    Returns:
    -------
    pl.DataFrame
        SUMMARY_SCHEMA DataFrame with agg values.
    """
    _ = fps
    return (
        df.group_by(MEASURE, GROUP)
        .agg(
            [
                # pl.col(VALUE).mean().alias("mean"),  # noqa: ERA001
                # pl.col(VALUE).std(ddof=0).alias("std"),  # noqa: ERA001
                # pl.col(VALUE).min().alias("min"),  # noqa: ERA001
                # pl.col(VALUE).quantile(0.25).alias("Q1"),  # noqa: ERA001
                # pl.col(VALUE).median().alias("median"),  # noqa: ERA001
                # pl.col(VALUE).quantile(0.75).alias("Q3"),  # noqa: ERA001
                # pl.col(VALUE).max().alias("max"),  # noqa: ERA001
                pl.col(VALUE).sum().alias("sum"),
            ],
        )
        .unpivot(index=[MEASURE, GROUP], variable_name=AGG, value_name=VALUE)
    )


def agg_behaviour(df: pl.DataFrame, fps: float) -> pl.DataFrame:
    """Summarize behavioural data: bout frequency and duration stats.

    Parameters
    ----------
    df : pl.DataFrame
        ANALYSIS_SCHEMA DataFrame (value is boolean 0/1 for behaviour presence).
    fps : float
        Frames per second for duration conversion.

    Returns:
    -------
    pl.DataFrame
        SUMMARY_SCHEMA DataFrame with bout_freq, bout_dur_* statistics.
    """
    results = []

    for (measure, grp), frame_group in df.group_by([MEASURE, GROUP]):
        vect = frame_group.sort(FRAME).select(VALUE).to_series()
        bouts = vect2bouts(vect == 1)
        dur_sec = bouts.select(pl.col(DUR) / fps).to_series()

        bout_freq = float(bouts.height)
        if len(dur_sec) == 0:
            dur_sec = pl.Series([0.0])

        stats = {
            GROUP: grp,
            MEASURE: measure,
            "bout_freq": bout_freq,
            "bout_dur_total": float(dur_sec.sum()),
            # "bout_dur_mean": float(dur_sec.mean()),  # noqa: ERA001
            # "bout_dur_std": float(dur_sec.std(ddof=0)) if len(dur_sec) > 1 else 0.0,  # noqa: E501, ERA001
            # "bout_dur_min": float(dur_sec.min()),  # noqa: ERA001
            # "bout_dur_Q1": float(dur_sec.quantile(0.25)),  # noqa: ERA001
            # "bout_dur_median": float(dur_sec.median()),  # noqa: ERA001
            # "bout_dur_Q3": float(dur_sec.quantile(0.75)),  # noqa: ERA001
            # "bout_dur_max": float(dur_sec.max()),  # noqa: ERA001
        }
        results.append(stats)

    if not results:
        return pl.DataFrame(schema=SUMMARY_SCHEMA)

    return pl.DataFrame(results).unpivot(
        index=[MEASURE, GROUP],
        variable_name=AGG,
        value_name=VALUE,
    )


def make_binned(
    analysis_df: pl.DataFrame,
    fps: float,
    bins_: list[float],
    summary_func: Callable[[pl.DataFrame, float], pl.DataFrame],
) -> pl.DataFrame:
    """Bin analysis data and apply summary function.

    Parameters
    ----------
    analysis_df : pl.DataFrame
        ANALYSIS_SCHEMA DataFrame.
    fps : float
        Frames per second.
    bins_ : list[float]
        Bin edges in seconds.
    summary_func : Callable
        Summary function (agg_quantitative or agg_behaviour).

    Returns:
    -------
    pl.DataFrame
        BINNED_SCHEMA DataFrame.
    """
    timestamps = analysis_df.select(
        (pl.col(FRAME) / fps).alias("timestamp"),
    ).to_series()
    t_max = float(timestamps.max())

    bins_arr = np.asarray(bins_, dtype=np.float64)
    if np.min(bins_arr) > 0:
        bins_arr = np.append(0.0, bins_arr)
    if np.max(bins_arr) < t_max:
        bins_arr = np.append(bins_arr, t_max)

    labels = bins_arr[1:]

    ts_np = timestamps.to_numpy()
    indices = np.searchsorted(bins_arr, ts_np, side="right") - 1
    indices = np.clip(indices, 0, len(labels) - 1)
    bin_col = pl.Series(BIN_SEC, labels[indices], dtype=pl.Float64)

    df_binned = analysis_df.with_columns(bin_col)

    results = []
    for (bin_val,), group in df_binned.group_by(BIN_SEC):
        summary = summary_func(group.drop(BIN_SEC), fps)
        summary = summary.with_columns(pl.lit(bin_val).alias(BIN_SEC))
        results.append(summary)

    if not results:
        return pl.DataFrame(schema=BINNED_SCHEMA)

    return pl.concat(results).select([BIN_SEC, MEASURE, GROUP, AGG, VALUE])


def summary_binned_quantitative(
    analysis_df: pl.DataFrame,
    name: str,
    fps: float,
    bins_ls: list[float],
    cbins_ls: list[float],
) -> list[AnalysisResult]:
    """Generate binned summary for quantitative data."""
    return summary_binned(
        analysis_df=analysis_df,
        name=name,
        fps=fps,
        summary_func=agg_quantitative,
        bins_ls=bins_ls,
        cbins_ls=cbins_ls,
    )


def summary_binned_behaviour(
    analysis_df: pl.DataFrame,
    name: str,
    fps: float,
    bins_sec_ls: list[float],
    custom_bins_sec_ls: list[float],
) -> list[AnalysisResult]:
    """Generate binned summary for behavioural data including latency."""
    results = summary_binned(
        analysis_df=analysis_df,
        name=name,
        fps=fps,
        summary_func=agg_behaviour,
        bins_ls=bins_sec_ls,
        cbins_ls=custom_bins_sec_ls,
    )
    latency_rows = _compute_latency(analysis_df, fps)
    if latency_rows:
        latency_df = pl.DataFrame(latency_rows, schema=SUMMARY_SCHEMA)
        results[0].result = pl.concat([results[0].result, latency_df])
    return results


def _compute_latency(analysis_df: pl.DataFrame, fps: float) -> list[dict]:
    """Compute latency: time to first positive value per (measure, group)."""
    latency_rows = []
    for (measure, grp), group in analysis_df.group_by([MEASURE, GROUP]):
        sorted_group = group.sort(FRAME)
        vect = sorted_group.select(VALUE).to_series()
        frame = sorted_group.select(FRAME).to_series()
        latency_val = -1.0
        if vect.sum() > 0:
            first_idx = (vect == 1).arg_true().item(0)
            latency_val = float(frame[first_idx]) / fps
        latency_rows.append(
            {
                GROUP: grp,
                MEASURE: measure,
                AGG: "latency",
                VALUE: latency_val,
            },
        )
    return latency_rows


def summary_binned(  # noqa: PLR0913
    analysis_df: pl.DataFrame,
    name: str,
    fps: float,
    summary_func: Callable[[pl.DataFrame, float], pl.DataFrame],
    bins_ls: list[float],
    cbins_ls: list[float],
) -> list[AnalysisResult]:
    """Return AnalysisResult objects for summary + binned DataFrames.

    Parameters
    ----------
    analysis_df : pl.DataFrame
        ANALYSIS_SCHEMA DataFrame.
    name : str
        Experiment name.
    fps : float
        Frames per second.
    summary_func : Callable
        Summary function (agg_quantitative or agg_behaviour).
    bins_ls : list[float]
        Standard bin sizes in seconds.
    cbins_ls : list[float]
        Custom bin sizes in seconds.
    """
    min_frame = analysis_df.select(FRAME).min().item()
    analysis_df = analysis_df.with_columns(
        (pl.col(FRAME) - min_frame).alias(FRAME),
    )

    summary_df = summary_func(analysis_df, fps)
    results = [
        AnalysisResult(
            relative_path=Path(SUMMARY) / f"{name}.{DF_IO_FORMAT}",
            result=summary_df,
            save_func=lambda fp, obj: write_df(obj, fp, SUMMARY_SCHEMA),
        ),
    ]

    timestamps = analysis_df.select(
        (pl.col(FRAME) / fps).alias("timestamp"),
    ).to_series()
    t_max = float(timestamps.max())

    for bin_sec in bins_ls:
        bins = np.arange(0, t_max + bin_sec, bin_sec).tolist()
        binned_df = make_binned(analysis_df, fps, bins, summary_func)
        results.append(
            AnalysisResult(
                relative_path=Path(f"{BINNED}_{bin_sec:g}") / f"{name}.{DF_IO_FORMAT}",
                result=binned_df,
                save_func=lambda fp, obj: write_df(obj, fp, BINNED_SCHEMA),
            ),
        )

    if cbins_ls:
        binned_df = make_binned(analysis_df, fps, cbins_ls, summary_func)
        results.append(
            AnalysisResult(
                relative_path=Path(f"{BINNED}_{CUSTOM}") / f"{name}.{DF_IO_FORMAT}",
                result=binned_df,
                save_func=lambda fp, obj: write_df(obj, fp, BINNED_SCHEMA),
            ),
        )

    return results
