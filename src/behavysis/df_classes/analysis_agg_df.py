"""Aggregated analysis DataFrames for summary and binned statistics."""

from collections.abc import Callable
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from behavysis.df_classes.analysis_df import AnalysisDf
from behavysis.df_classes.behav_df import BehavScoredDf
from behavysis.utils.df_mixin import DFMixin

SUMMARY = "summary"
BINNED = "binned"
PLOT = "plot"
CUSTOM = "custom"


class AnalysisSummaryIN(Enum):
    INDIVIDUALS = "individuals"
    MEASURES = "measures"


class AnalysisSummaryCN(Enum):
    AGGS = "aggs"


class AnalysisBinnedIN(Enum):
    BIN_SEC = "bin_sec"


class AnalysisBinnedCN(Enum):
    INDIVIDUALS = "individuals"
    MEASURES = "measures"
    AGGS = "aggs"


class AnalysisSummaryDf(DFMixin):
    NULLABLE = False
    IN = AnalysisSummaryIN
    CN = AnalysisSummaryCN

    @classmethod
    def agg_quantitative(cls, analysis_df: pd.DataFrame, fps: float) -> pd.DataFrame:
        """Summarize quantitative data: mean, std, min, Q1, median, Q3, max, sum."""
        summary_df_ls = np.zeros(analysis_df.shape[1], dtype="object")
        for i, col in enumerate(analysis_df.columns):
            vect = analysis_df[col]
            vect = pd.Series([0]) if len(vect) == 0 else vect
            vect = vect.astype(np.float64)
            summary_df_ls[i] = (
                pd.Series(
                    {
                        "mean": np.nanmean(vect),
                        "std": np.nanstd(vect),
                        "min": np.nanmin(vect),
                        "Q1": np.nanquantile(vect, q=0.25),
                        "median": np.nanmedian(vect),
                        "Q3": np.nanquantile(vect, q=0.75),
                        "max": np.nanmax(vect),
                        "sum": np.nansum(vect),
                    },
                    name=col,
                )
                .to_frame()
                .T
            )
        summary_df = pd.concat(summary_df_ls, axis=0)
        summary_df.index = analysis_df.columns
        return cls.clean_and_validate(summary_df)

    @classmethod
    def agg_behavs(cls, analysis_df: pd.DataFrame, fps: float) -> pd.DataFrame:
        """Summarize behavioral data: bout frequency and duration stats."""
        summary_df_ls = np.zeros(analysis_df.shape[1], dtype="object")
        for i, col in enumerate(analysis_df.columns):
            vect = analysis_df[col]
            bouts = BehavScoredDf.vect2bouts_df(vect == 1)["dur"] / fps
            bout_freq = len(bouts)
            bouts = pd.Series([0]) if len(bouts) == 0 else bouts.astype(np.float64)
            summary_df_ls[i] = (
                pd.Series(
                    {
                        "bout_freq": bout_freq,
                        "bout_dur_total": np.nansum(bouts),
                        "bout_dur_mean": np.nanmean(bouts),
                        "bout_dur_std": np.nanstd(bouts),
                        "bout_dur_min": np.nanmin(bouts),
                        "bout_dur_Q1": np.nanquantile(bouts, q=0.25),
                        "bout_dur_median": np.nanmedian(bouts),
                        "bout_dur_Q3": np.nanquantile(bouts, q=0.75),
                        "bout_dur_max": np.nanmax(bouts),
                    },
                    name=col,
                )
                .to_frame()
                .T
            )
        summary_df = pd.concat(summary_df_ls, axis=0)
        summary_df.index = analysis_df.columns
        return cls.clean_and_validate(summary_df)


class AnalysisBinnedDf(DFMixin):
    NULLABLE = False
    IN = AnalysisBinnedIN
    CN = AnalysisBinnedCN

    @classmethod
    def make_binned(
        cls,
        analysis_df: pd.DataFrame,
        fps: float,
        bins_: list,
        summary_func: Callable[[pd.DataFrame, float], pd.DataFrame],
    ) -> pd.DataFrame:
        """Bin analysis data and apply summary function."""
        timestamps = analysis_df.index.get_level_values("frame") / fps
        bins = np.asarray(bins_)
        bins = np.append(0, bins) if np.min(bins) > 0 else bins
        t_max = np.max(timestamps)
        bins = np.append(bins, t_max) if np.max(bins) < t_max else bins

        bin_sec = pd.cut(x=timestamps, bins=bins, labels=bins[1:], include_lowest=True)
        grouped_df = analysis_df.groupby(bin_sec)

        in_names = [e.value for e in AnalysisSummaryDf.IN]
        cn_names = [e.value for e in cls.CN]

        binned_df = grouped_df.apply(
            lambda x: (
                summary_func(x, fps)
                .unstack(in_names)
                .reorder_levels(cn_names)
                .sort_index(level=in_names)
            )
        )
        return cls.clean_and_validate(binned_df)

    @classmethod
    def make_binned_plot(
        cls, binned_df: pd.DataFrame, dst_fp: Path, agg_column: str
    ) -> None:
        """Plot binned data over time."""
        in_names = [e.value for e in AnalysisSummaryDf.IN]
        binned_stacked_df = (
            binned_df.stack(in_names)[agg_column].rename("value").reset_index()
        )
        g = sns.relplot(
            data=binned_stacked_df,
            x=cls.IN.BIN_SEC.value,
            y="value",
            hue="measures",
            col="individuals",
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

    @classmethod
    def summary_binned_quantitative(
        cls,
        analysis_df: pd.DataFrame,
        dst_dir: Path,
        name: str,
        fps: float,
        bins_ls: list,
        cbins_ls: list,
    ) -> None:
        """Generate binned summary for quantitative data."""
        return cls.summary_binned(
            analysis_df=analysis_df,
            dst_dir=dst_dir,
            name=name,
            fps=fps,
            summary_func=AnalysisSummaryDf.agg_quantitative,
            agg_column="mean",
            bins_ls=bins_ls,
            cbins_ls=cbins_ls,
        )

    @classmethod
    def summary_binned_behavs(
        cls,
        analysis_df: pd.DataFrame,
        dst_dir: Path,
        name: str,
        fps: float,
        bins_ls: list,
        cbins_ls: list,
    ) -> None:
        """Generate binned summary for behavioral data including latency."""
        cls.summary_binned(
            analysis_df=analysis_df,
            dst_dir=dst_dir,
            name=name,
            fps=fps,
            summary_func=AnalysisSummaryDf.agg_behavs,
            agg_column="bout_dur_total",
            bins_ls=bins_ls,
            cbins_ls=cbins_ls,
        )
        # Add latency (time from start to first bout)
        latency_df_ls = np.zeros(analysis_df.shape[1], dtype="object")
        for i, col in enumerate(analysis_df.columns):
            vect = analysis_df[col]
            vect = pd.Series([0]) if len(vect) == 0 else vect.astype(np.float64)
            index = vect.index.get_level_values(AnalysisDf.IN.FRAME.value) / fps
            latency_df_ls[i] = (
                pd.Series(
                    {"latency": index[vect == 1][0] if np.any(vect == 1) else -1},
                    name=col,
                )
                .to_frame()
                .T
            )
        latency_df = pd.concat(latency_df_ls, axis=0)
        latency_df.index = analysis_df.columns
        latency_df = AnalysisSummaryDf.clean_and_validate(latency_df)

        summary_df = AnalysisSummaryDf.agg_behavs(analysis_df, fps)
        summary_df = pd.concat([summary_df, latency_df], axis=1)
        summary_df = AnalysisSummaryDf.clean_and_validate(summary_df)

        summary_fp = dst_dir / SUMMARY / f"{name}.{cls.IO}"
        AnalysisSummaryDf.write(summary_df, summary_fp)

    @classmethod
    def summary_binned(
        cls,
        analysis_df: pd.DataFrame,
        dst_dir: Path,
        name: str,
        fps: float,
        summary_func: Callable[[pd.DataFrame, float], pd.DataFrame],
        agg_column: str,
        bins_ls: list,
        cbins_ls: list,
    ) -> None:
        """Generate binned summaries for standard and custom bins."""
        # Offset frames index to start from 0
        index_df = analysis_df.index.to_frame(index=False)
        frame_name = AnalysisDf.IN.FRAME.value
        index_df[frame_name] = index_df[frame_name] - index_df[frame_name].iloc[0]
        analysis_df.index = pd.MultiIndex.from_frame(index_df)

        # Generate summary
        summary_fp = dst_dir / SUMMARY / f"{name}.{cls.IO}"
        summary_df = summary_func(analysis_df, fps)
        AnalysisSummaryDf.write(summary_df, summary_fp)

        timestamps = analysis_df.index.get_level_values(AnalysisDf.IN.FRAME.value) / fps

        # Standard bins
        for bin_sec in bins_ls:
            binned_fp = dst_dir / f"{BINNED}_{bin_sec}" / f"{name}.{cls.IO}"
            binned_plot_fp = dst_dir / f"{BINNED}_{bin_sec}_{PLOT}" / f"{name}.png"
            bins = np.arange(0, np.max(timestamps) + bin_sec, bin_sec)
            binned_df = cls.make_binned(analysis_df, fps, bins, summary_func)
            cls.write(binned_df, binned_fp)
            cls.make_binned_plot(binned_df, binned_plot_fp, agg_column)

        # Custom bins
        if cbins_ls:
            binned_fp = dst_dir / f"{BINNED}_{CUSTOM}" / f"{name}.{cls.IO}"
            binned_plot_fp = dst_dir / f"{BINNED}_{CUSTOM}_{PLOT}" / f"{name}.png"
            binned_df = cls.make_binned(analysis_df, fps, cbins_ls, summary_func)
            cls.write(binned_df, binned_fp)
            cls.make_binned_plot(binned_df, binned_plot_fp, agg_column)
