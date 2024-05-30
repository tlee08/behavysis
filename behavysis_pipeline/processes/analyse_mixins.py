"""
Functions have the following format:

Parameters
----------
dlc_fp : str
    The DLC dataframe filepath of the experiment to analyse.
analysis_dir : str
    The analysis directory path.
configs_fp : str
    the experiment's JSON configs file.

Returns
-------
str
    The outcome of the process.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from behavysis_core.constants import (
    AGG_ANALYSIS_COLUMN_NAMES,
    ANALYSIS_COLUMN_NAMES,
    ANALYSIS_INDEX_NAME,
)
from behavysis_core.data_models.experiment_configs import ExperimentConfigs
from behavysis_core.mixins.behav_mixin import BehavMixin
from behavysis_core.mixins.df_io_mixin import DFIOMixin

#####################################################################
#               ANALYSIS API FUNCS
#####################################################################


class AnalyseMixin:
    """__summary__"""

    @staticmethod
    def get_configs(
        configs: ExperimentConfigs,
    ) -> tuple[
        float,
        float,
        float,
        float,
        list,
        list,
    ]:
        """
        _summary_

        Parameters
        ----------
        configs : Configs
            _description_

        Returns
        -------
        tuple[ float, float, float, float, list, list, ]
            _description_
        """
        return (
            configs.auto.formatted_vid.fps,
            configs.auto.formatted_vid.width_px,
            configs.auto.formatted_vid.height_px,
            configs.auto.px_per_mm,
            configs.get_ref(configs.user.analyse.bins_sec),
            configs.get_ref(configs.user.analyse.custom_bins_sec),
        )

    @staticmethod
    def init_df(frame_vect: pd.Series | pd.Index) -> pd.DataFrame:
        """
        Returning a frame-by-frame analysis_df with the frame number (according to original video)
        as the MultiIndex index, relative to the first element of frame_vect.
        Note that that the frame number can thus begin on a non-zero number.

        Parameters
        ----------
        frame_vect : pd.Series | pd.Index
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        return pd.DataFrame(
            index=pd.Index(frame_vect, name=ANALYSIS_INDEX_NAME),
            columns=pd.MultiIndex.from_tuples((), names=ANALYSIS_COLUMN_NAMES),
        )

    @staticmethod
    def check_df(df: pd.DataFrame) -> None:
        """
        Checks whether the dataframe is in the correct format for the keypoints functions.

        Checks that:

        - There are no null values.
        - The column levels are correct.
        - The index levels are correct.
        """
        # Checking for null values
        assert (
            not df.isnull().values.any()
        ), "The dataframe contains null values. Be sure to run interpolate_points first."
        # Checking that the index levels are correct
        assert (
            df.index.name == ANALYSIS_INDEX_NAME
        ), f"The index level is incorrect. They should be {ANALYSIS_INDEX_NAME}"
        # Checking that the column levels are correct
        assert (
            df.columns.names == ANALYSIS_COLUMN_NAMES
        ), f"The column levels are incorrect. They should be {ANALYSIS_COLUMN_NAMES}."

    @staticmethod
    def read_feather(fp: str) -> pd.DataFrame:
        """
        Reading feather file.
        """
        # Reading
        df = DFIOMixin.read_feather(fp)
        # Checking
        AnalyseMixin.check_df(df)
        # Returning
        return df

    @staticmethod
    def make_location_scatterplot(
        analysis_df: pd.DataFrame, roi_df: pd.DataFrame, out_fp, measure: str
    ):
        """
        Expects analysis_df index levels to be (frame,),
        and column levels to be (individual, measure).
        """
        analysis_stacked_df = analysis_df.stack(level="individuals").reset_index(
            "individuals"
        )
        g = sns.relplot(
            data=analysis_stacked_df,
            x="x",
            y="y",
            hue=measure,
            col="individuals",
            kind="scatter",
            col_wrap=2,
            height=4,
            aspect=1,
            alpha=0.8,
            linewidth=0,
            marker=".",
            s=10,
            legend=True,
        )
        # Invert the y axis
        g.axes[0].invert_yaxis()
        # Adding region definition (from roi_df) to the plot
        first_row = pd.DataFrame(roi_df.iloc[0]).T
        roi_df = pd.concat((roi_df, first_row), ignore_index=True)
        for ax in g.axes:
            sns.lineplot(
                data=roi_df,
                x="x",
                y="y",
                color=(1, 0, 0),
                linewidth=1,
                marker="+",
                markeredgecolor=(1, 0, 0),
                markeredgewidth=2,
                markersize=5,
                estimator=None,
                sort=False,
                legend=False,
                ax=ax,
            )
        # Setting fig titles and labels
        g.set_titles(col_template="{col_name}")
        g.figure.subplots_adjust(top=0.85)
        g.figure.suptitle("Spatial position", fontsize=12)
        # Saving fig
        os.makedirs(os.path.split(out_fp)[0], exist_ok=True)
        g.savefig(out_fp)
        g.figure.clf()

    @staticmethod
    def pt_in_roi(pt: pd.Series, roi_df: pd.DataFrame) -> bool:
        """__summary__"""
        # Counting crossings over edge in region when point is translated to the right
        crossings = 0
        # To loop back to the first point at the end
        first_roi_pt = pd.DataFrame(roi_df.iloc[0]).T
        roi_df = pd.concat((roi_df, first_roi_pt), axis=0, ignore_index=True)
        # For each edge
        for i in range(roi_df.shape[0] - 1):
            # Getting corner points of edge
            c1 = roi_df.iloc[i]
            c2 = roi_df.iloc[i + 1]
            # Getting whether point-y is between corners-y
            y_between = (c1["y"] > pt["y"]) != (c2["y"] > pt["y"])
            # Getting whether point-x is to the left (less than) the intersection of corners-x
            x_left_of = (
                pt["x"]
                < (c2["x"] - c1["x"]) * (pt["y"] - c1["y"]) / (c2["y"] - c1["y"])
                + c1["x"]
            )
            if y_between and x_left_of:
                crossings += 1
        return crossings % 2 == 1

    @staticmethod
    def pt_in_roi_df(
        dlc_df: pd.DataFrame, roi_df: pd.DataFrame, indivs: list[str], bpts: list[str]
    ) -> pd.DataFrame:
        """__summary__"""
        res_df = AnalyseMixin.init_df(dlc_df.index)
        idx = pd.IndexSlice
        for indiv in indivs:
            res_df[(indiv, "x")] = (
                dlc_df.loc[:, idx[indiv, bpts, "x"]].mean(axis=1).values
            )
            res_df[(indiv, "y")] = (
                dlc_df.loc[:, idx[indiv, bpts, "y"]].mean(axis=1).values
            )
            # Determining if the indiv is outside the boundaries
            res_df[(indiv, "in_roi")] = (
                res_df[indiv]
                .apply(lambda pt: AnalyseMixin.pt_in_roi(pt, roi_df), axis=1)
                .astype(np.int8)
            )
        return res_df


class AggAnalyse:
    """__summary__"""

    @staticmethod
    def agg_quantitative(analysis_df: pd.DataFrame, fps: float) -> pd.DataFrame:
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
        summary_df = pd.DataFrame()
        # Getting summary stats for each individual
        for name, vect in analysis_df.items():
            # Handling edge case where columns are empty
            vect = np.array([0]) if vect.shape[0] == 0 else vect
            # Setting columns to type float
            vect = vect.astype(np.float64)
            # Aggregating stats
            summary_df_i = (
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
                    name=name,
                )
                .to_frame()
                .T
            )
            summary_df = pd.concat([summary_df, summary_df_i], axis=0)
        summary_df.index = analysis_df.columns
        summary_df.columns.name = AGG_ANALYSIS_COLUMN_NAMES[2]
        # Returning summary_df
        return summary_df

    @staticmethod
    def agg_behavs(analysis_df: pd.DataFrame, fps: float) -> pd.DataFrame:
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
        summary_df = pd.DataFrame()
        # Getting summary stats for each individual
        for name, vect in analysis_df.items():
            # Getting duration of each behav bout
            bouts = BehavMixin.vect_2_bouts(vect == 1)["dur"]
            # Getting bout frequency (before it is overwritten if empty)
            bout_freq = bouts.shape[0]
            # Handling edge case where bouts is empty
            bouts = np.array([0]) if bouts.shape[0] == 0 else bouts
            # Setting bouts to type float
            bouts = bouts.astype(np.float64)
            # Aggregating stats
            summary_df_i = (
                pd.Series(
                    {
                        "bout_freq": bout_freq,
                        "bout_dur_total": np.nansum(bouts) / fps,
                        "bout_dur_mean": np.nanmean(bouts) / fps,
                        "bout_dur_std": np.nanstd(bouts) / fps,
                        "bout_dur_min": np.nanmin(bouts) / fps,
                        "bout_dur_Q1": np.nanquantile(bouts, q=0.25) / fps,
                        "bout_dur_median": np.nanmedian(bouts) / fps,
                        "bout_dur_Q3": np.nanquantile(bouts, q=0.75) / fps,
                        "bout_dur_max": np.nanmax(bouts) / fps,
                    },
                    name=name,
                )
                .to_frame()
                .T
            )
            summary_df = pd.concat([summary_df, summary_df_i], axis=0)
        summary_df.index = analysis_df.columns
        summary_df.columns.name = AGG_ANALYSIS_COLUMN_NAMES[2]
        # Returning summary_df
        return summary_df

    @staticmethod
    def make_binned(
        analysis_df: pd.DataFrame,
        fps: float,
        bins: list,
        summary_func: Callable[[pd.DataFrame, float], pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Generates the binned data and line graph for the given analysis_df, and given bin_sec.
        The aggregated statistics are very similar to the summary data.
        """
        # For each column, displays the mean of each binned group.
        timestamps = analysis_df.index.get_level_values("frame") / fps
        # Ensuring all bins are included (start frame and end frame)
        bins = np.append(0, bins) if np.min(bins) > 0 else bins
        t_max = np.max(timestamps)
        bins = np.append(bins, t_max) if np.max(bins) < t_max else bins
        # Making binned data
        bin_sec = pd.cut(timestamps, bins=bins, labels=bins[1:], include_lowest=True)
        grouped_df = analysis_df.groupby(bin_sec)
        binned_df = grouped_df.apply(
            lambda x: summary_func(x, fps)
            .unstack(ANALYSIS_COLUMN_NAMES)
            .reorder_levels(AGG_ANALYSIS_COLUMN_NAMES)
            .sort_index(level=ANALYSIS_COLUMN_NAMES)
        )
        binned_df.index.name = "bin_sec"
        # returning binned_df
        return binned_df

    @staticmethod
    def make_binned_plot(
        binned_df: pd.DataFrame,
        out_fp: str,
        agg_column: str,
    ):
        """
        _summary_
        """
        # Making binned_df long
        binned_stacked_df = (
            binned_df.stack(ANALYSIS_COLUMN_NAMES)[agg_column]
            .rename("value")
            .reset_index()
        )
        # Plotting line graph
        g = sns.relplot(
            data=binned_stacked_df,
            x="bin_sec",
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
        os.makedirs(os.path.split(out_fp)[0], exist_ok=True)
        g.savefig(out_fp)
        g.figure.clf()

    @staticmethod
    def summary_binned_quantitative(
        analysis_df: pd.DataFrame,
        out_dir: str,
        name: str,
        fps: float,
        bins_ls: Optional[list],
        cbins_ls: Optional[list],
    ) -> str:
        """
        _summary_
        """
        return AggAnalyse.summary_binned(
            analysis_df=analysis_df,
            out_dir=out_dir,
            name=name,
            fps=fps,
            summary_func=AggAnalyse.agg_quantitative,
            agg_column="mean",
            bins_ls=bins_ls,
            cbins_ls=cbins_ls,
        )

    @staticmethod
    def summary_binned_behavs(
        analysis_df: pd.DataFrame,
        out_dir: str,
        name: str,
        fps: float,
        bins_ls: Optional[list],
        cbins_ls: Optional[list],
    ) -> str:
        """
        _summary_
        """
        return AggAnalyse.summary_binned(
            analysis_df=analysis_df,
            out_dir=out_dir,
            name=name,
            fps=fps,
            summary_func=AggAnalyse.agg_behavs,
            agg_column="bout_dur_total",
            bins_ls=bins_ls,
            cbins_ls=cbins_ls,
        )

    @staticmethod
    def summary_binned(
        analysis_df: pd.DataFrame,
        out_dir: str,
        name: str,
        fps: float,
        summary_func: Callable[[pd.DataFrame, float], pd.DataFrame],
        agg_column: str,
        bins_ls: Optional[list],
        cbins_ls: Optional[list],
    ) -> str:
        """
        _summary_
        """
        outcome = ""
        # Summarising analysis_df
        summary_fp = os.path.join(out_dir, "summary", f"{name}.feather")
        summary_df = summary_func(analysis_df, fps)
        DFIOMixin.write_feather(summary_df, summary_fp)
        # Getting timestamps index
        timestamps = analysis_df.index.get_level_values("frame") / fps
        # Binning analysis_df
        for bin_sec in bins_ls:
            # Making filepaths
            binned_fp = os.path.join(out_dir, f"binned_{bin_sec}", f"{name}.feather")
            binned_plot_fp = os.path.join(
                out_dir, f"binned_{bin_sec}_plot", f"{name}.png"
            )
            # Making binned df
            bins = np.arange(0, np.max(timestamps) + bin_sec, bin_sec)
            binned_df = AggAnalyse.make_binned(analysis_df, fps, bins, summary_func)
            DFIOMixin.write_feather(binned_df, binned_fp)
            # Making binned plots
            AggAnalyse.make_binned_plot(binned_df, binned_plot_fp, agg_column)
        # Custom binning analysis_df
        if cbins_ls:
            # Making filepaths
            binned_fp = os.path.join(out_dir, "binned_custom", f"{name}.feather")
            binned_plot_fp = os.path.join(out_dir, "binned_custom_plot", f"{name}.png")
            # Making binned df
            binned_df = AggAnalyse.make_binned(analysis_df, fps, cbins_ls, summary_func)
            DFIOMixin.write_feather(binned_df, binned_fp)
            # Making binned plots
            AggAnalyse.make_binned_plot(binned_df, binned_plot_fp, agg_column)
        # Returning outcome
        return outcome
