import os
from enum import Enum
from typing import Callable

import numpy as np
import pandas as pd
import seaborn as sns

from behavysis.df_classes.analysis_df import AnalysisDf
from behavysis.df_classes.behav_df import BehavScoredDf
from behavysis.df_classes.df_mixin import DFMixin
from behavysis.utils.misc_utils import enum2list, enum2tuple

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
        """
        Generates the summarised data across the entire period, including mean,
        std, min, Q1, median, Q3, and max.
        Used for quantitative numeric data.

        Params:
            analysis_df: pd.DataFrame
                _description_

        Returns:
        str
            The outcome string.
        """
        # Getting summary stats for each individual
        summary_df_ls = np.zeros(analysis_df.shape[1], dtype="object")
        for i, col in enumerate(analysis_df.columns):
            # Getting column vector of individual-measure
            vect = analysis_df[col]
            # Handling edge case where columns are empty
            vect = np.array([0]) if vect.shape[0] == 0 else vect
            # Setting columns to type float
            vect = vect.astype(np.float64)
            # Aggregating stats
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
                    },
                    name=col,
                )
                .to_frame()
                .T
            )
        # Concatenating summary_df_ls, setting index, and cleaning
        summary_df = pd.concat(summary_df_ls, axis=0)
        summary_df.index = analysis_df.columns
        summary_df = cls.basic_clean(summary_df)
        return summary_df

    @classmethod
    def agg_behavs(cls, analysis_df: pd.DataFrame, fps: float) -> pd.DataFrame:
        """
        Generates the summarised data across the entire period, including number of bouts,
        and mean, std, min, Q1, median, Q3, and max duration of bouts.
        Used for boolean behavs classification data.

        Parameters
        ----------
        analysis_df : pd.DataFrame
            _description_
        Returns
        -------
        str
            The outcome string.
        """
        # Getting summary stats for each individual
        summary_df_ls = np.zeros(analysis_df.shape[1], dtype="object")
        for i, col in enumerate(analysis_df.columns):
            # Getting column vector of individual-measure
            vect = analysis_df[col]
            # Getting duration of each behav bout
            bouts = BehavScoredDf.vect2bouts_df(vect == 1)["dur"]
            # Converting bouts duration from frames to seconds
            bouts = bouts / fps
            # Getting bout frequency (before it is overwritten if empty)
            bout_freq = bouts.shape[0]
            # Handling edge case where bouts is empty
            bouts = np.array([0]) if bouts.shape[0] == 0 else bouts
            # Setting bouts to type float
            bouts = bouts.astype(np.float64)
            # Aggregating stats
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
        # Concatenating summary_df_ls, setting index, and cleaning
        summary_df = pd.concat(summary_df_ls, axis=0)
        summary_df.index = analysis_df.columns
        summary_df = cls.basic_clean(summary_df)
        return summary_df


class AnalysisBinnedDf(DFMixin):
    """__summary__"""

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
        """
        Generates the binned data and line graph for the given analysis_df, and given bin_sec.
        The aggregated statistics are very similar to the summary data.
        """
        # For each column, displays the mean of each binned group.
        timestamps = analysis_df.index.get_level_values("frame") / fps
        # Ensuring all bins are included (start frame and end frame)
        bins = np.asarray(bins_)
        bins = np.append(0, bins) if np.min(bins) > 0 else bins
        t_max = np.max(timestamps)
        bins = np.append(bins, t_max) if np.max(bins) < t_max else bins
        # Making binned data
        bin_sec = pd.cut(x=timestamps, bins=bins, labels=bins[1:], include_lowest=True)  # type: ignore
        grouped_df = analysis_df.groupby(bin_sec)
        binned_df = grouped_df.apply(
            lambda x: summary_func(x, fps)
            .unstack(enum2tuple(AnalysisSummaryDf.IN))
            .reorder_levels(enum2list(cls.CN))
            .sort_index(level=enum2tuple(AnalysisSummaryDf.IN))
        )
        # Cleaning (sets index and column names) and checking
        binned_df = cls.basic_clean(binned_df)
        return binned_df

    @classmethod
    def make_binned_plot(
        cls,
        binned_df: pd.DataFrame,
        dst_fp: str,
        agg_column: str,
    ):
        """
        _summary_
        """
        # Making binned_df long
        binned_stacked_df = binned_df.stack(enum2tuple(AnalysisSummaryDf.IN))[agg_column].rename("value").reset_index()
        # Plotting line graph
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
        # Setting fig titles and labels
        g.set_titles(col_template="{col_name}")
        g.figure.subplots_adjust(top=0.85)
        g.figure.suptitle("Binned data", fontsize=12)
        # Saving fig
        os.makedirs(os.path.dirname(dst_fp), exist_ok=True)
        g.savefig(dst_fp)
        g.figure.clf()

    @classmethod
    def summary_binned_quantitative(
        cls,
        analysis_df: pd.DataFrame,
        dst_dir: str,
        name: str,
        fps: float,
        bins_ls: list,
        cbins_ls: list,
    ) -> str:
        """
        _summary_
        """
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
        dst_dir: str,
        name: str,
        fps: float,
        bins_ls: list,
        cbins_ls: list,
    ) -> str:
        """
        _summary_
        """
        return cls.summary_binned(
            analysis_df=analysis_df,
            dst_dir=dst_dir,
            name=name,
            fps=fps,
            summary_func=AnalysisSummaryDf.agg_behavs,
            agg_column="bout_dur_total",
            bins_ls=bins_ls,
            cbins_ls=cbins_ls,
        )

    @classmethod
    def summary_binned(
        cls,
        analysis_df: pd.DataFrame,
        dst_dir: str,
        name: str,
        fps: float,
        summary_func: Callable[[pd.DataFrame, float], pd.DataFrame],
        agg_column: str,
        bins_ls: list,
        cbins_ls: list,
    ) -> str:
        """
        _summary_
        """
        outcome = ""
        # Offsetting the frames index to start from 0
        # (i.e. when the experiment commenced, rather than when the recording started)
        index_df = analysis_df.index.to_frame(index=False)
        frame_name = AnalysisDf.IN.FRAME.value
        index_df[frame_name] = index_df[frame_name] - index_df[frame_name].iloc[0]
        analysis_df.index = pd.MultiIndex.from_frame(index_df)
        # Summarising analysis_df
        summary_fp = os.path.join(dst_dir, SUMMARY, f"{name}.{cls.IO}")
        summary_csv_fp = os.path.join(dst_dir, f"{SUMMARY}_csv", f"{name}.csv")
        summary_df = summary_func(analysis_df, fps)
        AnalysisSummaryDf.write(summary_df, summary_fp)
        AnalysisSummaryDf.write_csv(summary_df, summary_csv_fp)
        # Getting timestamps index
        timestamps = analysis_df.index.get_level_values(AnalysisDf.IN.FRAME.value) / fps
        # Binning analysis_df
        for bin_sec in bins_ls:
            # Making filepaths
            binned_fp = os.path.join(dst_dir, f"{BINNED}_{bin_sec}", f"{name}.{cls.IO}")
            binned_csv_fp = os.path.join(dst_dir, f"{BINNED}_{bin_sec}_csv", f"{name}.csv")
            binned_plot_fp = os.path.join(dst_dir, f"{BINNED}_{bin_sec}_{PLOT}", f"{name}.png")
            # Making binned df
            bins = np.arange(0, np.max(timestamps) + bin_sec, bin_sec)
            binned_df = cls.make_binned(analysis_df, fps, bins, summary_func)
            cls.write(binned_df, binned_fp)
            cls.write_csv(binned_df, binned_csv_fp)
            # Making binned plots
            cls.make_binned_plot(binned_df, binned_plot_fp, agg_column)
        # Custom binning analysis_df
        if cbins_ls:
            # Making filepaths
            binned_fp = os.path.join(dst_dir, f"{BINNED}_{CUSTOM}", f"{name}.{cls.IO}")
            binned_csv_fp = os.path.join(dst_dir, f"{BINNED}_{CUSTOM}_csv", f"{name}.csv")
            binned_plot_fp = os.path.join(dst_dir, f"{BINNED}_{CUSTOM}_{PLOT}", f"{name}.png")
            # Making binned df
            binned_df = cls.make_binned(analysis_df, fps, cbins_ls, summary_func)
            cls.write(binned_df, binned_fp)
            cls.write_csv(binned_df, binned_csv_fp)
            # Making binned plots
            cls.make_binned_plot(binned_df, binned_plot_fp, agg_column)
        return outcome
