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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from behavysis_core.data_models.experiment_configs import ExperimentConfigs
from behavysis_core.mixins.behaviour_mixin import BehaviourMixin
from behavysis_core.mixins.df_io_mixin import DFIOMixin
from behavysis_core.mixins.io_mixin import IOMixin
from behavysis_core.mixins.keypoints_mixin import KeypointsMixin
from behavysis_core.utils.constants import (
    ANALYSIS_COLUMN_NAMES,
    ANALYSIS_INDEX_NAMES,
    BODYCENTRE,
    SINGLE_COL,
)

#####################################################################
#               ANALYSIS API FUNCS
#####################################################################


class Analyse:
    @staticmethod
    def thigmotaxis(
        dlc_fp: str,
        analysis_dir: str,
        configs_fp: str,
    ) -> str:
        """
        Determines the frames when the subject is in thigmotaxis.

        Takes DLC data as input and returns the following analysis output:

        - A feather file with the thigmotaxis data columns for each video frame (row)
        - A png of the scatterplot of the subject's x-y position in every frame, coloured by whether
        it was in thigmotaxis.
        - A png of the bivariate histogram distribution of the subject's x-y position for all frames,
        coloured by whether it was in thigmotaxis.
        """
        outcome = ""
        name = IOMixin.get_name(dlc_fp)
        out_dir = os.path.join(analysis_dir, Analyse.thigmotaxis.__name__)
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        fps, _, _, px_per_mm, bins_ls, custom_bins_ls = get_analysis_configs(configs)
        configs_filt = configs.user.analyse.thigmotaxis
        thresh_mm = float(configs_filt.thresh_mm)
        # Calculating more parameters
        thresh_px = thresh_mm / px_per_mm
        # Loading in dataframe
        dlc_df = KeypointsMixin.clean_headings(DFIOMixin.read_feather(dlc_fp))
        # Getting indivs and bpts list
        indivs, _ = KeypointsMixin.get_headings(dlc_df)

        # Getting average corner coordinates. Assumes arena does not move.
        tl = dlc_df[(SINGLE_COL, "TopLeft")].apply(np.nanmean)
        tr = dlc_df[(SINGLE_COL, "TopRight")].apply(np.nanmean)
        bl = dlc_df[(SINGLE_COL, "BottomLeft")].apply(np.nanmean)
        br = dlc_df[(SINGLE_COL, "BottomRight")].apply(np.nanmean)
        # Making boundary functions
        top = hline_factory(tl, tr)
        bottom = hline_factory(bl, br)
        left = vline_factory(tl, bl)
        right = vline_factory(tr, br)

        analysis_df = init_fbf_analysis_df(dlc_df.index, fps)
        dlc_df.index = analysis_df.index
        for indiv in indivs:
            indiv_x = dlc_df[(indiv, BODYCENTRE, "x")]
            indiv_y = dlc_df[(indiv, BODYCENTRE, "y")]
            # Determining if the indiv is outside of the boundaries (with the thresh_px buffer)
            analysis_df[(indiv, "thigmotaxis")] = (
                (indiv_y <= top(indiv_x) + thresh_px)
                | (indiv_y >= bottom(indiv_x) - thresh_px)
                | (indiv_x <= left(indiv_y) + thresh_px)
                | (indiv_x >= right(indiv_y) - thresh_px)
            ).astype(np.int8)
        # Saving analysis_df
        fbf_fp = os.path.join(out_dir, "fbf", f"{name}.feather")
        DFIOMixin.write_feather(analysis_df, fbf_fp)

        # Generating scatterplot
        # Adding bodypoint x and y coords
        for indiv in indivs:
            analysis_df[(indiv, "x")] = dlc_df[(indiv, BODYCENTRE, "x")]
            analysis_df[(indiv, "y")] = dlc_df[(indiv, BODYCENTRE, "y")]
        # making corners_df
        corners_df = pd.DataFrame([tl, tr, bl, br])
        plot_fp = os.path.join(out_dir, "scatter_plot", f"{name}.png")
        make_location_scatterplot(analysis_df, corners_df, plot_fp, "thigmotaxis")

        # Summarising and binning analysis_df
        make_summary_binned(
            DFIOMixin.read_feather(fbf_fp), out_dir, name, bins_ls, custom_bins_ls, True
        )
        return outcome

    @staticmethod
    def center_crossing(
        dlc_fp: str,
        analysis_dir: str,
        configs_fp: str,
    ) -> str:
        """
        Determines the frames when the subject is in center (reverse of thigmotaxis).

        Takes DLC data as input and returns the following analysis output:

        - A feather file with the thigmotaxis data columns for each video frame (row)
        - A png of the scatterplot of the subject's x-y position in every frame, coloured by whether
        it was in thigmotaxis.
        - A png of the bivariate histogram distribution of the subject's x-y position for all
        frames, coloured by whether it was in thigmotaxis.
        """
        outcome = ""
        name = IOMixin.get_name(dlc_fp)
        out_dir = os.path.join(analysis_dir, Analyse.center_crossing.__name__)
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        fps, _, _, px_per_mm, bins_ls, custom_bins_ls = get_analysis_configs(configs)
        configs_filt = configs.user.analyse.center_crossing
        thresh_mm = float(configs_filt.thresh_mm)
        # Calculating more parameters
        thresh_px = thresh_mm / px_per_mm
        # Loading in dataframe
        dlc_df = KeypointsMixin.clean_headings(DFIOMixin.read_feather(dlc_fp))
        # Getting indivs and bpts list
        indivs, _ = KeypointsMixin.get_headings(dlc_df)

        # Getting average corner coordinates. NOTE: assumes arena does not move.
        tl = dlc_df[(SINGLE_COL, "TopLeft")].apply(np.nanmean)
        tr = dlc_df[(SINGLE_COL, "TopRight")].apply(np.nanmean)
        bl = dlc_df[(SINGLE_COL, "BottomLeft")].apply(np.nanmean)
        br = dlc_df[(SINGLE_COL, "BottomRight")].apply(np.nanmean)
        # Making boundary functions
        top = hline_factory(tl, tr)
        bottom = hline_factory(bl, br)
        left = vline_factory(tl, bl)
        right = vline_factory(tr, br)

        analysis_df = init_fbf_analysis_df(dlc_df.index, fps)
        dlc_df.index = analysis_df.index
        for indiv in indivs:
            indiv_x = dlc_df[(indiv, BODYCENTRE, "x")]
            indiv_y = dlc_df[(indiv, BODYCENTRE, "y")]
            # Determining if the indiv is outside of the boundaries (with the thresh_px buffer)
            analysis_df[(indiv, "in_center")] = (
                (indiv_y > top(indiv_x) + thresh_px)
                & (indiv_y < bottom(indiv_x) - thresh_px)
                & (indiv_x > left(indiv_y) + thresh_px)
                & (indiv_x < right(indiv_y) - thresh_px)
            ).astype(np.int8)
        # Saving analysis_df
        fbf_fp = os.path.join(out_dir, "fbf", f"{name}.feather")
        DFIOMixin.write_feather(analysis_df, fbf_fp)

        # Generating scatterplot
        # Adding bodypoint x and y coords
        for indiv in indivs:
            analysis_df[(indiv, "x")] = dlc_df[(indiv, BODYCENTRE, "x")]
            analysis_df[(indiv, "y")] = dlc_df[(indiv, BODYCENTRE, "y")]
        # making corners_df
        corners_df = pd.DataFrame([tl, tr, bl, br])
        plot_fp = os.path.join(out_dir, "scatter_plot", f"{name}.png")
        make_location_scatterplot(analysis_df, corners_df, plot_fp, "in_center")

        # Summarising and binning analysis_df
        make_summary_binned(
            DFIOMixin.read_feather(fbf_fp), out_dir, name, bins_ls, custom_bins_ls, True
        )
        return outcome

    @staticmethod
    def speed(
        dlc_fp: str,
        analysis_dir: str,
        configs_fp: str,
    ) -> str:
        """
        Determines the speed of the subject in each frame.

        Takes DLC data as input and returns the following analysis output:

        - a feather file with the following columns for each video frame (row).
        - a feather file with the summary statistics (sum, mean, std, min, median, Q1, median, Q3,
        max) for DeltaMMperSec, and DeltaMMperSecSmoothed
        """
        outcome = ""
        name = IOMixin.get_name(dlc_fp)
        out_dir = os.path.join(analysis_dir, Analyse.speed.__name__)
        # Calculating the deltas (changes in body position) between each frame for the subject
        configs = ExperimentConfigs.read_json(configs_fp)
        fps, _, _, px_per_mm, bins_ls, custom_bins_ls = get_analysis_configs(configs)
        configs_filt = configs.user.analyse.speed
        smoothing_sec = configs_filt.smoothing_sec
        # Calculating more parameters
        smoothing_frames = int(smoothing_sec * fps)
        # Loading in dataframe
        dlc_df = KeypointsMixin.clean_headings(DFIOMixin.read_feather(dlc_fp))
        # Getting indivs and bpts list
        indivs, _ = KeypointsMixin.get_headings(dlc_df)
        # Checking body-centre bodypart exists
        KeypointsMixin.check_bpts_exist(dlc_df, [BODYCENTRE])

        # Calculating speed of subject for each frame
        analysis_df = init_fbf_analysis_df(dlc_df.index, fps)
        dlc_df.index = analysis_df.index
        for indiv in indivs:
            # Making a rolling window of 3?? maybe 4 frames for average body-centre
            # Otherwise jitter contributes to movement
            jitter_frames = 3
            smoothed_xy_df = dlc_df.rolling(
                window=jitter_frames, center=True, min_periods=1
            ).agg(np.nanmean)
            delta_x = smoothed_xy_df[(indiv, BODYCENTRE, "x")].diff()
            delta_y = smoothed_xy_df[(indiv, BODYCENTRE, "y")].diff()
            delta = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))
            analysis_df[(indiv, "SpeedMMperSec")] = (delta / px_per_mm) * fps
            analysis_df[(indiv, "SpeedMMperSecSmoothed")] = (
                analysis_df[(indiv, "SpeedMMperSec")]
                .rolling(window=smoothing_frames, min_periods=1)
                .agg(np.nanmean)
            )
        # Saving analysis_df
        fbf_fp = os.path.join(out_dir, "fbf", f"{name}.feather")
        DFIOMixin.write_feather(analysis_df, fbf_fp)

        # Summarising and binning analysis_df
        make_summary_binned(
            DFIOMixin.read_feather(fbf_fp),
            out_dir,
            name,
            bins_ls,
            custom_bins_ls,
            False,
        )
        return outcome

    @staticmethod
    def social_distance(
        dlc_fp: str,
        analysis_dir: str,
        configs_fp: str,
    ) -> str:
        """
        Determines the speed of the subject in each frame.

        Takes DLC data as input and returns the following analysis output:

        - a feather file with the following columns for each video frame (row).
        - a feather file with the summary statistics (sum, mean, std, min, median, Q1, median, Q3,
        max) for DeltaMMperSec, and DeltaMMperSecSmoothed
        """
        outcome = ""
        name = IOMixin.get_name(dlc_fp)
        out_dir = os.path.join(analysis_dir, Analyse.social_distance.__name__)
        # Calculating the deltas (changes in body position) between each frame for the subject
        configs = ExperimentConfigs.read_json(configs_fp)
        fps, _, _, px_per_mm, bins_ls, custom_bins_ls = get_analysis_configs(configs)
        configs_filt = configs.user.analyse.social_distance
        smoothing_sec = configs_filt.smoothing_sec
        # Calculating more parameters
        smoothing_frames = int(smoothing_sec * fps)
        # Loading in dataframe
        dlc_df = KeypointsMixin.clean_headings(DFIOMixin.read_feather(dlc_fp))
        KeypointsMixin.check_bpts_exist(dlc_df, [BODYCENTRE])
        # Getting indivs and bpts list
        indivs, _ = KeypointsMixin.get_headings(dlc_df)

        # Calculating speed of subject for each frame
        analysis_df = init_fbf_analysis_df(dlc_df.index, fps)
        dlc_df.index = analysis_df.index
        # Assumes there are only two individuals
        indiv_a = indivs[0]
        indiv_b = indivs[1]
        bpt = "Nose"
        # Getting distances between each individual
        dist_x = dlc_df[(indiv_a, bpt, "x")] - dlc_df[(indiv_b, bpt, "x")]
        dist_y = dlc_df[(indiv_a, bpt, "y")] - dlc_df[(indiv_b, bpt, "y")]
        dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))
        # Adding mm distance to saved analysis_df table
        analysis_df[(f"{indiv_a}_{indiv_b}", "DistMM")] = dist / px_per_mm
        analysis_df[(f"{indiv_a}_{indiv_b}", "DistMMSmoothed")] = (
            analysis_df[(f"{indiv_a}_{indiv_b}", "DistMM")]
            .rolling(window=smoothing_frames, min_periods=1)
            .agg(np.nanmean)
        )
        # Saving analysis_df
        fbf_fp = os.path.join(out_dir, "fbf", f"{name}.feather")
        DFIOMixin.write_feather(analysis_df, fbf_fp)

        # Summarising and binning analysis_df
        make_summary_binned(
            DFIOMixin.read_feather(fbf_fp),
            out_dir,
            name,
            bins_ls,
            custom_bins_ls,
            False,
        )
        return outcome

    @staticmethod
    def freezing(
        dlc_fp: str,
        analysis_dir: str,
        configs_fp: str,
    ) -> str:
        """
        Determines the frames in which the subject is frozen.

        "Frozen" is defined as not moving outside of a radius of `threshold_radius_mm`, and only
        includes bouts that last longer than `window_sec` spent seconds.

        NOTE: method is "greedy" because it looks at a freezing bout from earliest possible frame.

        Takes DLC data as input and returns the following analysis output:

        - a feather file with the following columns for each video frame (row).
        - a feather file with the summary statistics (sum, mean, std, min, median, Q1, median,
        Q3, max) for DeltaMMperSec, and DeltaMMperSecSmoothed
        - Each row `is_frozen`, and bout number.
        """
        outcome = ""
        name = IOMixin.get_name(dlc_fp)
        out_dir = os.path.join(analysis_dir, Analyse.freezing.__name__)
        # Calculating the deltas (changes in body position) between each frame for the subject
        configs = ExperimentConfigs.read_json(configs_fp)
        fps, _, _, px_per_mm, bins_ls, custom_bins_ls = get_analysis_configs(configs)
        configs_filt = configs.user.analyse.freezing
        window_sec = configs_filt.window_sec
        thresh_mm = configs_filt.thresh_mm
        smoothing_sec = configs_filt.smoothing_sec
        # Calculating more parameters
        thresh_px = thresh_mm / px_per_mm
        smoothing_frames = int(smoothing_sec * fps)
        window_frames = int(np.round(fps * window_sec, 0))
        # Loading in dataframe
        dlc_df = KeypointsMixin.clean_headings(DFIOMixin.read_feather(dlc_fp))
        # Getting indivs and bpts list
        indivs, bpts = KeypointsMixin.get_headings(dlc_df)

        # Calculating speed of subject for each frame
        analysis_df = init_fbf_analysis_df(dlc_df.index, fps)
        dlc_df.index = analysis_df.index
        for indiv in indivs:
            temp_df = pd.DataFrame(index=analysis_df.index)
            # Calculating frame-by-frame delta distances for current bpt
            for bpt in bpts:
                # Getting x and y changes
                delta_x = dlc_df[(indiv, bpt, "x")].diff()
                delta_y = dlc_df[(indiv, bpt, "y")].diff()
                # Getting Euclidean distance between frames for bpt
                delta = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))
                # Converting from px to mm
                temp_df[f"{bpt}_dist"] = delta
                # Smoothing
                temp_df[f"{bpt}_dist"] = (
                    temp_df[f"{bpt}_dist"]
                    .rolling(window=smoothing_frames, min_periods=1)
                    .agg(np.nanmean)
                )
            # If ALL bodypoints do not leave `thresh_px`
            analysis_df[(indiv, "freezing")] = temp_df.apply(
                lambda x: pd.Series(np.all(x < thresh_px)), axis=1
            ).astype(np.int8)

            # Getting start, stop, and duration of each freezing behav bout
            freezingbouts_df = BehaviourMixin.vect_2_bouts(
                analysis_df[(indiv, "freezing")] == 1
            )
            # For each freezing bout, if there is less than window_frames, tehn
            # it is not actually freezing
            for _, row in freezingbouts_df.iterrows():
                if row["dur"] < window_frames:
                    analysis_df.loc[row["start"] : row["stop"], (indiv, "freezing")] = 0
        # Saving analysis_df
        fbf_fp = os.path.join(out_dir, "fbf", f"{name}.feather")
        DFIOMixin.write_feather(analysis_df, fbf_fp)

        # Summarising and binning analysis_df
        make_summary_binned(
            DFIOMixin.read_feather(fbf_fp), out_dir, name, bins_ls, custom_bins_ls, True
        )
        return outcome

    @staticmethod
    def in_roi(
        dlc_fp: str,
        analysis_dir: str,
        configs_fp: str,
    ) -> str:
        """
        Determines the frames in which the subject's nose is inside the cage.

        Takes DLC data as input and returns the following analysis output:

        - a feather file with the following columns for each video frame (row).
        - a feather file with the summary statistics (sum, mean, std, min, median, Q1, median,
        Q3, max) for DeltaMMperSec, and DeltaMMperSecSmoothed
        - Each row `is_frozen`, and bout number.
        """
        outcome = ""
        name = IOMixin.get_name(dlc_fp)
        out_dir = os.path.join(analysis_dir, Analyse.in_roi.__name__)
        # Calculating the deltas (changes in body position) between each frame for the subject
        configs = ExperimentConfigs.read_json(configs_fp)
        fps, _, _, px_per_mm, bins_ls, custom_bins_ls = get_analysis_configs(configs)
        configs_filt = configs.user.analyse.in_roi
        thresh_mm = configs_filt.thresh_mm
        tl_label = configs_filt.roi_top_left
        tr_label = configs_filt.roi_top_right
        bl_label = configs_filt.roi_bottom_left
        br_label = configs_filt.roi_bottom_right
        bpts = configs_filt.bodyparts
        # Calculating more parameters
        thresh_px = thresh_mm / px_per_mm
        # Loading in dataframe
        dlc_df = KeypointsMixin.clean_headings(DFIOMixin.read_feather(dlc_fp))
        # Getting indivs and bpts list
        indivs, bpts = KeypointsMixin.get_headings(dlc_df)
        # Checking body-centre bodypart exists
        KeypointsMixin.check_bpts_exist(dlc_df, ["Nose"])

        # Getting average corner coordinates. Assumes arena does not move.
        tl = dlc_df[(SINGLE_COL, tl_label)].apply(np.nanmean)
        tr = dlc_df[(SINGLE_COL, tr_label)].apply(np.nanmean)
        bl = dlc_df[(SINGLE_COL, bl_label)].apply(np.nanmean)
        br = dlc_df[(SINGLE_COL, br_label)].apply(np.nanmean)
        # Making boundary functions
        top = hline_factory(tl, tr)
        bottom = hline_factory(bl, br)
        left = vline_factory(tl, bl)
        right = vline_factory(tr, br)

        analysis_df = init_fbf_analysis_df(dlc_df.index, fps)
        dlc_df.index = analysis_df.index
        idx = pd.IndexSlice
        for indiv in indivs:
            indiv_x = dlc_df.loc[:, idx[indiv, bpts, "x"]].apply(np.nanmean, axis=1)
            indiv_y = dlc_df.loc[:, idx[indiv, bpts, "y"]].apply(np.nanmean, axis=1)
            # Determining if the indiv is inside of the box region (with the thresh_px buffer)
            analysis_df[(indiv, "in_roi")] = (
                (indiv_y >= top(indiv_x) - thresh_px)
                & (indiv_y <= bottom(indiv_x) + thresh_px)
                & (indiv_x >= left(indiv_y) - thresh_px)
                & (indiv_x <= right(indiv_y) + thresh_px)
            ).astype(np.int8)
        # Saving analysis_df
        fbf_fp = os.path.join(out_dir, "fbf", f"{name}.feather")
        DFIOMixin.write_feather(analysis_df, fbf_fp)

        # Generating scatterplot
        # Adding bodypoint x and y coords
        for indiv in indivs:
            analysis_df[(indiv, "x")] = dlc_df.loc[:, idx[indiv, bpts, "x"]].apply(
                np.nanmean, axis=1
            )
            analysis_df[(indiv, "y")] = dlc_df.loc[:, idx[indiv, bpts, "y"]].apply(
                np.nanmean, axis=1
            )
        # making corners_df
        corners_df = pd.DataFrame([tl, tr, bl, br])
        plot_fp = os.path.join(out_dir, "scatter_plot", f"{name}.png")
        make_location_scatterplot(analysis_df, corners_df, plot_fp, "in_roi")

        # Summarising and binning analysis_df
        make_summary_binned(
            DFIOMixin.read_feather(fbf_fp), out_dir, name, bins_ls, custom_bins_ls, True
        )
        return outcome


def get_analysis_configs(
    configs: ExperimentConfigs,
) -> tuple[
    int,
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
    tuple[ int, float, float, float, list, list, ]
        _description_
    """
    return (
        configs.auto.formatted_vid.fps,
        configs.auto.formatted_vid.width_px,
        configs.auto.formatted_vid.height_px,
        configs.auto.px_per_mm,
        configs.user.analyse.bins_sec,
        configs.user.analyse.custom_bins_sec,
    )


def init_fbf_analysis_df(frame_vect: pd.Series | pd.Index, fps: int) -> pd.DataFrame:
    """
    Returning a frame-by-frame analysis_df with the frame number (according to original video)
    and timestamps as the MultiIndex index, relative to the first element of frame_vect.
    Note that that the frame number can thus begin on a non-zero number.

    Parameters
    ----------
    frame_vect : pd.Series | pd.Index
        _description_
    fps : int
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    return pd.DataFrame(
        index=pd.MultiIndex.from_arrays(
            (frame_vect, (frame_vect - frame_vect[0]) / fps), names=ANALYSIS_INDEX_NAMES
        ),
        columns=pd.MultiIndex.from_tuples((), names=ANALYSIS_COLUMN_NAMES),
    )


def make_location_scatterplot(
    analysis_df: pd.DataFrame, corners_df: pd.DataFrame, out_fp, measure: str
):
    """
    Expects analysis_df index to be (frame, timestamp), and columns to be (individual, measure).

    Parameters
    ----------
    analysis_df : pd.DataFrame
        _description_
    corners_df : pd.DataFrame
        _description_
    out_fp : _type_
        _description_
    measure : str
        _description_
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
    # Invert the y axis and adding arena corners to the plot
    g.axes[0].invert_yaxis()
    for ax in g.axes:
        sns.scatterplot(
            data=corners_df,
            x="x",
            y="y",
            marker="+",
            color=(1, 0, 0),
            s=50,
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


def _make_summary_quantitative(
    analysis_df: pd.DataFrame,
    out_fp: str,
) -> str:
    """
    Generates the summarised data across the entire period, including mean,
    std, min, Q1, median, Q3, and max.
    Used for quantitative numeric data.

    Params:
        TODO

    Returns:
        The outcome string.
    """
    outcome = ""
    summary_df = pd.DataFrame()
    # Getting summary stats for each individual
    for column_name, column_vals in analysis_df.items():
        indiv_measure_summary = (
            pd.Series(
                {
                    "mean": np.nanmean(column_vals.astype(float)),
                    "std": np.nanstd(column_vals.astype(float)),
                    "min": np.nanmin(column_vals.astype(float)),
                    "Q1": np.nanquantile(column_vals.astype(float), q=0.25),
                    "median": np.nanmedian(column_vals.astype(float)),
                    "Q3": np.nanquantile(column_vals.astype(float), q=0.75),
                    "max": np.nanmax(column_vals.astype(float)),
                },
                name=column_name,
            )
            .to_frame()
            .transpose()
        )
        summary_df = pd.concat([summary_df, indiv_measure_summary], axis=0)
    summary_df.index = analysis_df.columns
    DFIOMixin.write_feather(summary_df, out_fp)
    return outcome


def _make_summary_behaviour(analysis_df: pd.DataFrame, out_fp: str) -> str:
    """
    Generates the summarised data across the entire period, including number of bouts,
    and mean, std, min, Q1, median, Q3, and max duration of bouts.
    Used for boolean behaviour classification data.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        _description_
    out_fp : str
        _description_

    Returns
    -------
    str
        The outcome string.
    """
    outcome = ""
    summary_df = pd.DataFrame()
    # Getting summary stats for each individual
    for column_name, column_vals in analysis_df.items():
        # Getting start, stop, and duration of each behav bout
        bouts_df = BehaviourMixin.vect_2_bouts(column_vals == 1)
        bouts = bouts_df["dur"]
        # Handling edge case where bouts is empty
        if bouts_df.shape[0] == 0:
            bouts = np.array([0])
        measure_summary_i = (
            pd.Series(
                {
                    "bout_freq": bouts_df.shape[0],
                    "bout_dur_mean": np.nanmean(bouts.astype(float)),
                    "bout_dur_std": np.nanstd(bouts.astype(float)),
                    "bout_dur_min": np.nanmin(bouts.astype(float)),
                    "bout_dur_Q1": np.nanquantile(bouts.astype(float), q=0.25),
                    "bout_dur_median": np.nanmedian(bouts.astype(float)),
                    "bout_dur_Q3": np.nanquantile(bouts.astype(float), q=0.75),
                    "bout_dur_max": np.nanmax(bouts.astype(float)),
                },
                name=column_name,
            )
            .to_frame()
            .transpose()
        )
        summary_df = pd.concat([summary_df, measure_summary_i], axis=0)
    summary_df.index = analysis_df.columns
    DFIOMixin.write_feather(summary_df, out_fp)
    return outcome


def _make_binned(
    analysis_df: pd.DataFrame,
    out_fp: str,
    bins: list,
) -> str:
    """
    Generates the binned data and line graph for the given analysis_df, and given bin_sec.

    # TODO - should this be user-changeable?
    # TODO - for behaviour (binary), make a bout frequency stat (binned) and mean bout time (binned)

    Parameters
    ----------
    analysis_df : pd.DataFrame
        _description_
    out_fp : str
        _description_
    bins : list
        _description_

    Returns
    -------
    str
        _description_
    """
    # For each column, displays the mean of each binned group.
    outcome = ""
    timestamps = analysis_df.index.get_level_values("timestamp")
    # Ensuring all bins are included (start frame and end frame)
    if np.min(bins) > 0:  # If 0 is not included
        bins = np.append(0, bins)
    if np.max(bins) < np.max(timestamps):  # If end timestamp is not included
        bins = np.append(bins, np.max(timestamps))
    # Making binned data
    bin_sec = pd.cut(timestamps, bins=bins, labels=bins[1:], include_lowest=True)
    binned_df = analysis_df.assign(bin_sec=bin_sec).groupby("bin_sec").agg(np.nanmean)
    # Writing binned_df to file
    DFIOMixin.write_feather(binned_df, out_fp)
    return outcome


def _make_binned_plot(
    binned_df: pd.DataFrame,
    out_fp: str,
    is_bool: bool,
) -> str:
    """
    _summary_

    Parameters
    ----------
    binned_df : pd.DataFrame
        _description_
    out_fp : str
        _description_
    is_bool : bool
        _description_

    Returns
    -------
    str
        _description_
    """
    outcome = ""
    # Making binned_df long
    binned_stacked_df = (
        binned_df.stack(["individuals", "measures"])
        .reset_index()
        .rename(columns={0: "value"})
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
    if is_bool:
        g.set(ylim=(0, 1))
    # Saving fig
    os.makedirs(os.path.split(out_fp)[0], exist_ok=True)
    g.savefig(out_fp)
    g.figure.clf()
    # Returning outcome
    return outcome


def make_summary_binned(
    analysis_df: pd.DataFrame,
    out_dir: str,
    name: str,
    bins_ls: list,
    custom_bins_ls: list,
    is_bool: bool,
) -> str:
    """
    _summary_

    Parameters
    ----------
    analysis_df : pd.DataFrame
        _description_
    out_dir : str
        _description_
    name : str
        _description_
    bins_ls : list
        _description_
    custom_bins_ls : list
        _description_
    is_bool : bool
        _description_

    Returns
    -------
    str
        _description_
    """
    outcome = ""
    # Summarising analysis_df
    summary_fp = os.path.join(out_dir, "summary", f"{name}.feather")
    if is_bool:
        outcome += _make_summary_behaviour(analysis_df, summary_fp)
    else:
        outcome += _make_summary_quantitative(analysis_df, summary_fp)
    # Getting timestamps index
    timestamps = analysis_df.index.get_level_values("timestamp")
    # Binning analysis_df
    for bin_sec in bins_ls:
        binned_fp = os.path.join(out_dir, f"binned_{bin_sec}", f"{name}.feather")
        binned_plot_fp = os.path.join(out_dir, f"binned_{bin_sec}_plot", f"{name}.png")
        # Making binned df
        bins = np.arange(np.min(timestamps), np.max(timestamps) + bin_sec, bin_sec)
        outcome += _make_binned(analysis_df, binned_fp, bins)
        # Making binned plots
        outcome += _make_binned_plot(
            DFIOMixin.read_feather(binned_fp), binned_plot_fp, is_bool
        )
    # Custom binning analysis_df
    binned_fp = os.path.join(out_dir, "binned_custom", f"{name}.feather")
    binned_plot_fp = os.path.join(out_dir, "binned_custom_plot", f"{name}.png")
    # Making binned df
    outcome += _make_binned(analysis_df, binned_fp, custom_bins_ls)
    # Making binned plots
    outcome += _make_binned_plot(
        DFIOMixin.read_feather(binned_fp), binned_plot_fp, is_bool
    )
    return outcome


def hline_factory(p1, p2):
    """
    Boundary function factories (x input to y).
    Making boundary line, given corner points. Expects input to be x
    m = (y2-y1)/(x2-x1)   &   y = m(x-x1) + b + y1
    """
    return lambda x: (p2["y"] - p1["y"]) / (p2["x"] - p1["x"]) * (x - p1["x"]) + p1["y"]


def vline_factory(p1, p2):
    """
    Boundary function factories (y input to x).
    Making boundary line, given corner points. Expects input to be y
    m = (x2-x1)/(y2-y1)   &   x = m(y-y1) + b + x1
    """
    return lambda y: (p2["x"] - p1["x"]) / (p2["y"] - p1["y"]) * (y - p1["y"]) + p1["x"]
