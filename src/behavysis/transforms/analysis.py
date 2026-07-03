"""Helper funcs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import seaborn as sns

from behavysis.constants import (
    BIN_SEC,
    BINNED,
    CUSTOM,
    DF_IO_FORMAT,
    INDIVIDUAL,
    MEASURE,
    PLOT,
    SUMMARY,
)
from behavysis.schemas import BINNED_SCHEMA, SUMMARY_SCHEMA, write_df

from .behaviour import vect2bouts

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


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
        SUMMARY_SCHEMA DataFrame with agg values:
        mean, std, min, Q1, median, Q3, max, sum.
    """
    _ = fps  # unused in quantitative agg
    return (
        df.group_by(INDIVIDUAL, MEASURE)
        .agg(
            [
                pl.col("value").mean().alias("mean"),
                pl.col("value").std(ddof=0).alias("std"),
                pl.col("value").min().alias("min"),
                pl.col("value").quantile(0.25).alias("Q1"),
                pl.col("value").median().alias("median"),
                pl.col("value").quantile(0.75).alias("Q3"),
                pl.col("value").max().alias("max"),
                pl.col("value").sum().alias("sum"),
            ],
        )
        .unpivot(index=[INDIVIDUAL, MEASURE], variable_name="agg", value_name="value")
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

    for (indiv, measure), group in df.group_by([INDIVIDUAL, MEASURE]):
        vect = group.sort("frame").select("value").to_series()
        bouts = vect2bouts(vect == 1)
        dur_sec = bouts.select(pl.col("dur") / fps).to_series()

        bout_freq = float(bouts.height)
        if len(dur_sec) == 0:
            dur_sec = pl.Series([0.0])

        stats = {
            INDIVIDUAL: indiv,
            MEASURE: measure,
            "bout_freq": bout_freq,
            "bout_dur_total": float(dur_sec.sum()),
            "bout_dur_mean": float(dur_sec.mean()),
            "bout_dur_std": float(dur_sec.std(ddof=0)) if len(dur_sec) > 1 else 0.0,
            "bout_dur_min": float(dur_sec.min()),
            "bout_dur_Q1": float(dur_sec.quantile(0.25)),
            "bout_dur_median": float(dur_sec.median()),
            "bout_dur_Q3": float(dur_sec.quantile(0.75)),
            "bout_dur_max": float(dur_sec.max()),
        }
        results.append(stats)

    if not results:
        return pl.DataFrame(schema=SUMMARY_SCHEMA)

    return pl.DataFrame(results).melt(
        id_vars=[INDIVIDUAL, MEASURE],
        variable_name="agg",
        value_name="value",
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
        (pl.col("frame") / fps).alias("timestamp"),
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

    return pl.concat(results).select([BIN_SEC, INDIVIDUAL, MEASURE, "agg", "value"])


def make_binned_plot(
    binned_df: pl.DataFrame,
    dst_fp: Path,
    agg_column: str,
) -> None:
    """Plot binned data over time using seaborn.

    Parameters
    ----------
    binned_df : pl.DataFrame
        BINNED_SCHEMA DataFrame.
    dst_fp : Path
        Filepath to save the plot.
    agg_column : str
        Aggregation column to plot (e.g. "mean", "bout_dur_total").
    """
    plot_df = binned_df.filter(pl.col("agg") == agg_column).to_pandas()

    g = sns.relplot(
        data=plot_df,
        x=BIN_SEC,
        y="value",
        hue=MEASURE,
        col=INDIVIDUAL,
        kind="line",
        height=4,
        aspect=1.5,
        alpha=0.5,
        marker="X",
        markersize=10,
        legend=True,
    )
    g.set_titles(col_template="{col_name}")
    g.figure.subplots_adjust(top=0.85)
    g.figure.suptitle("Binned data", fontsize=12)
    dst_fp.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(dst_fp)
    g.figure.clf()


def summary_binned_quantitative(
    analysis_df: pl.DataFrame,
    dst_dir: Path,
    name: str,
    fps: float,
    bins_ls: list[int],
    cbins_ls: list[int],
) -> None:
    """Generate binned summary for quantitative data."""
    summary_binned(
        analysis_df=analysis_df,
        dst_dir=dst_dir,
        name=name,
        fps=fps,
        summary_func=agg_quantitative,
        agg_column="mean",
        bins_ls=bins_ls,
        cbins_ls=cbins_ls,
    )


def summary_binned_behaviour(
    analysis_df: pl.DataFrame,
    dst_dir: Path,
    name: str,
    fps: float,
    bins_sec_ls: list[int],
    custom_bins_sec_ls: list[int],
) -> None:
    """Generate binned summary for behavioural data including latency."""
    summary_binned(
        analysis_df=analysis_df,
        dst_dir=dst_dir,
        name=name,
        fps=fps,
        summary_func=agg_behaviour,
        agg_column="bout_dur_total",
        bins_ls=bins_sec_ls,
        cbins_ls=custom_bins_sec_ls,
    )

    latency_rows = _compute_latency(analysis_df, fps)

    if latency_rows:
        latency_df = pl.DataFrame(latency_rows, schema=SUMMARY_SCHEMA)
        summary_df = agg_behaviour(analysis_df, fps)
        summary_df = pl.concat([summary_df, latency_df])
        summary_fp = dst_dir / SUMMARY / f"{name}.{DF_IO_FORMAT}"
        write_df(summary_df, summary_fp, SUMMARY_SCHEMA)


def _compute_latency(analysis_df: pl.DataFrame, fps: float) -> list[dict]:
    """Compute latency: time to first positive value per (individual, measure)."""
    latency_rows = []
    for (indiv, measure), group in analysis_df.group_by([INDIVIDUAL, MEASURE]):
        sorted_group = group.sort("frame")
        vect = sorted_group.select("value").to_series()
        frame = sorted_group.select("frame").to_series()
        latency_val = -1.0
        if vect.sum() > 0:
            first_idx = (vect == 1).arg_true().item()
            latency_val = float(frame[first_idx]) / fps
        latency_rows.append(
            {
                INDIVIDUAL: indiv,
                MEASURE: measure,
                "agg": "latency",
                "value": latency_val,
            },
        )
    return latency_rows


def summary_binned(
    analysis_df: pl.DataFrame,
    dst_dir: Path,
    name: str,
    fps: float,
    summary_func: Callable[[pl.DataFrame, float], pl.DataFrame],
    agg_column: str,
    bins_ls: list[int],
    cbins_ls: list[int],
) -> None:
    """Generate binned summaries for standard and custom bins.

    Parameters
    ----------
    analysis_df : pl.DataFrame
        ANALYSIS_SCHEMA DataFrame.
    dst_dir : Path
        Destination directory for outputs.
    name : str
        Experiment name.
    fps : float
        Frames per second.
    summary_func : Callable
        Summary function (agg_quantitative or agg_behaviour).
    agg_column : str
        Aggregation column for plotting.
    bins_ls : list[int]
        Standard bin sizes in seconds.
    cbins_ls : list[int]
        Custom bin sizes in seconds.
    """
    min_frame = analysis_df.select("frame").min().item()
    analysis_df = analysis_df.with_columns(
        (pl.col("frame") - min_frame).alias("frame"),
    )

    summary_fp = dst_dir / SUMMARY / f"{name}.{DF_IO_FORMAT}"
    summary_df = summary_func(analysis_df, fps)
    write_df(summary_df, summary_fp, SUMMARY_SCHEMA)

    timestamps = analysis_df.select(
        (pl.col("frame") / fps).alias("timestamp"),
    ).to_series()
    t_max = float(timestamps.max())

    for bin_sec in bins_ls:
        binned_fp = dst_dir / f"{BINNED}_{bin_sec}" / f"{name}.{DF_IO_FORMAT}"
        binned_plot_fp = dst_dir / f"{BINNED}_{bin_sec}_{PLOT}" / f"{name}.png"
        bins = np.arange(0, t_max + bin_sec, bin_sec).tolist()
        binned_df = make_binned(analysis_df, fps, bins, summary_func)
        write_df(binned_df, binned_fp, BINNED_SCHEMA)
        make_binned_plot(binned_df, binned_plot_fp, agg_column)

    if cbins_ls:
        binned_fp = dst_dir / f"{BINNED}_{CUSTOM}" / f"{name}.{DF_IO_FORMAT}"
        binned_plot_fp = dst_dir / f"{BINNED}_{CUSTOM}_{PLOT}" / f"{name}.png"
        binned_df = make_binned(analysis_df, fps, cbins_ls, summary_func)
        write_df(binned_df, binned_fp, BINNED_SCHEMA)
        make_binned_plot(binned_df, binned_plot_fp, agg_column)
