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

import numpy as np
import pandas as pd
from behavysis_core.constants import AnalysisCN, Coords, IndivColumns
from behavysis_core.data_models.experiment_configs import ExperimentConfigs
from behavysis_core.mixins.behav_df_mixin import BehavDfMixin
from behavysis_core.mixins.df_io_mixin import DFIOMixin
from behavysis_core.mixins.io_mixin import IOMixin
from behavysis_core.mixins.keypoints_df_mixin import KeypointsMixin
from pydantic import BaseModel, ConfigDict

from .analyse_mixin import AggAnalyse, AnalyseMixin

#####################################################################
#               ANALYSIS API FUNCS
#####################################################################


class Analyse:
    """__summary__"""

    @staticmethod
    def in_roi(
        dlc_fp: str,
        analysis_dir: str,
        configs_fp: str,
    ) -> str:
        """
        Determines the frames in which the subject is inside the cage (from average
        of given bodypoints).

        Takes DLC data as input and returns the following analysis output:

        - a feather file with the following columns for each video frame (row).
        - a feather file with the summary statistics (sum, mean, std, min, median, Q1, median,
        Q3, max) for DeltaMMperSec, and DeltaMMperSecSmoothed
        - Each row `is_frozen`, and bout number.
        """
        outcome = ""
        name = IOMixin.get_name(dlc_fp)
        f_name = Analyse.in_roi.__name__
        out_dir = os.path.join(analysis_dir, f_name)
        # Calculating the deltas (changes in body position) between each frame for the subject
        configs = ExperimentConfigs.read_json(configs_fp)
        fps, _, _, px_per_mm, bins_ls, cbins_ls = AnalyseMixin.get_configs(configs)
        configs_filt_ls = list(configs.user.analyse.in_roi)  # type: ignore
        # Loading in dataframe
        dlc_df = KeypointsMixin.clean_headings(KeypointsMixin.read_feather(dlc_fp))
        # Getting indivs list
        indivs, _ = KeypointsMixin.get_headings(dlc_df)
        # Making analysis_df
        analysis_df_ls = []
        roi_c_df_ls = []
        # For each roi, calculate the in-roi status of the subject
        x = Coords.X.value
        y = Coords.Y.value
        idx = pd.IndexSlice
        for configs_filt in configs_filt_ls:
            # Getting necessary config parameters
            configs_filt = Model_in_roi(**configs_filt)
            roi_name = configs.get_ref(configs_filt.roi_name)
            is_in = configs.get_ref(configs_filt.is_in)
            bpts = configs.get_ref(configs_filt.bodyparts)
            thresh_mm = configs.get_ref(configs_filt.thresh_mm)
            roi_corners = configs.get_ref(configs_filt.roi_corners)
            # Calculating more parameters
            thresh_px = thresh_mm / px_per_mm
            # Checking bodyparts and roi_corners exist
            KeypointsMixin.check_bpts_exist(dlc_df, bpts)
            KeypointsMixin.check_bpts_exist(dlc_df, roi_corners)
            # Getting average corner coordinates. Assumes arena does not move.
            roi_c_df = pd.DataFrame(
                [dlc_df[(IndivColumns.SINGLE.value, pt)].mean() for pt in roi_corners]
            ).drop(columns=["likelihood"])
            # Adjusting x-y to have a distance dilation/erosion from the points themselves
            roi_center = roi_c_df.mean()
            for i in roi_c_df.index:
                # Calculating angle from point to centre
                theta = np.arctan2(
                    roi_c_df.loc[i, y] - roi_center[y],
                    roi_c_df.loc[i, x] - roi_center[x],
                )
                # Getting x, y distances so point is `thresh_px` padded (away) from center
                roi_c_df.loc[i, x] = roi_c_df.loc[i, x] + (thresh_px * np.cos(theta))
                roi_c_df.loc[i, y] = roi_c_df.loc[i, y] + (thresh_px * np.sin(theta))
            # Making the res_df
            res_df = AnalyseMixin.init_df(dlc_df.index)
            # For each individual, getting the in-roi status
            for indiv in indivs:
                # Getting average body center (x, y) for each individual
                res_df[(indiv, x)] = (
                    dlc_df.loc[:, idx[indiv, bpts, x]].mean(axis=1).values  # type: ignore
                )
                res_df[(indiv, y)] = (
                    dlc_df.loc[:, idx[indiv, bpts, y]].mean(axis=1).values  # type: ignore
                )
                # Determining if the indiv body center is in the ROI
                res_df[(indiv, "in_roi")] = (
                    res_df[indiv]
                    .apply(lambda pt: AnalyseMixin.pt_in_roi(pt, roi_c_df), axis=1)
                    .astype(np.int8)
                )
            # Inverting in_roi status if is_in is False
            if not is_in:
                res_df.loc[:, idx[:, "in_roi"]] = ~res_df.loc[:, idx[:, "in_roi"]]  # type: ignore
            # Changing column MultiIndex names
            res_df.columns = res_df.columns.set_levels(  # type: ignore
                ["x", "y", f"in_roi_{roi_name}"], level=AnalysisCN.MEASURES.value
            )
            # Saving to analysis_df and roi_corners_df list
            analysis_df_ls.append(res_df.loc[:, idx[:, f"in_roi_{roi_name}"]])  # type: ignore
            roi_c_df_ls.append(roi_c_df)
        # Concatenating all analysis_df_ls and roi_corners_df_ls
        analysis_df = pd.concat(analysis_df_ls, axis=1)
        roi_c_df = pd.concat(roi_c_df_ls, keys=range(len(roi_c_df_ls)), names=["group"])
        roi_c_df = roi_c_df.reset_index(level="group")
        # Saving analysis_df
        fbf_fp = os.path.join(out_dir, "fbf", f"{name}.feather")
        DFIOMixin.write_feather(analysis_df, fbf_fp)
        # Generating scatterplot
        # First getting scatter_in_roi columns
        # TODO: any way to include all different "x", "y" to use, rather
        # than the last res_df?
        scatter_df = res_df.loc[:, idx[:, ["x", "y"]]]  # type: ignore
        for i in indivs:
            scatter_df[(i, "roi")] = analysis_df[i].apply(
                lambda x: "-".join(x.index[x == 1]), axis=1
            )
        # Making and saving scatterplot
        plot_fp = os.path.join(out_dir, "scatter_plot", f"{name}.png")
        AnalyseMixin.make_location_scatterplot(scatter_df, roi_c_df, plot_fp, "roi")
        # Summarising and binning analysis_df
        AggAnalyse.summary_binned_behavs(
            analysis_df,
            out_dir,
            name,
            fps,
            bins_ls,
            cbins_ls,
        )
        # Returning outcome
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
        f_name = Analyse.speed.__name__
        out_dir = os.path.join(analysis_dir, f_name)
        # Calculating the deltas (changes in body position) between each frame for the subject
        configs = ExperimentConfigs.read_json(configs_fp)
        fps, _, _, px_per_mm, bins_ls, cbins_ls = AnalyseMixin.get_configs(configs)
        configs_filt = Model_speed(**configs.user.analyse.speed)  # type: ignore
        bpts = configs.get_ref(configs_filt.bodyparts)
        smoothing_sec = configs.get_ref(configs_filt.smoothing_sec)
        # Calculating more parameters
        smoothing_frames = int(smoothing_sec * fps)

        # Loading in dataframe
        dlc_df = KeypointsMixin.clean_headings(KeypointsMixin.read_feather(dlc_fp))
        # Checking body-centre bodypart exists
        KeypointsMixin.check_bpts_exist(dlc_df, bpts)
        # Getting indivs and bpts list
        indivs, _ = KeypointsMixin.get_headings(dlc_df)

        # Calculating speed of subject for each frame
        analysis_df = AnalyseMixin.init_df(dlc_df.index)
        dlc_df.index = analysis_df.index
        idx = pd.IndexSlice
        for indiv in indivs:
            # Making a rolling window of 3(??) frames for average body-centre
            # Otherwise jitter contributes to movement
            jitter_frames = 3
            smoothed_xy_df = dlc_df.rolling(
                window=jitter_frames, center=True, min_periods=1
            ).agg(np.nanmean)
            delta_x = smoothed_xy_df.loc[:, idx[indiv, bpts, "x"]].mean(axis=1).diff()  # type: ignore
            delta_y = smoothed_xy_df.loc[:, idx[indiv, bpts, "y"]].mean(axis=1).diff()  # type: ignore
            delta = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))
            analysis_df[(indiv, "SpeedMMperSec")] = (delta / px_per_mm) * fps
            analysis_df[(indiv, "SpeedMMperSecSmoothed")] = (
                analysis_df[(indiv, "SpeedMMperSec")]
                .rolling(window=smoothing_frames, min_periods=1)
                .agg(np.nanmean)
            )
        # Backfilling the analysis_df (because of diff and rolling window)
        analysis_df = analysis_df.bfill()
        # Saving analysis_df
        fbf_fp = os.path.join(out_dir, "fbf", f"{name}.feather")
        DFIOMixin.write_feather(analysis_df, fbf_fp)

        # Summarising and binning analysis_df
        AggAnalyse.summary_binned_quantitative(
            analysis_df,
            out_dir,
            name,
            fps,
            bins_ls,
            cbins_ls,
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
        f_name = Analyse.social_distance.__name__
        out_dir = os.path.join(analysis_dir, f_name)
        # Calculating the deltas (changes in body position) between each frame for the subject
        configs = ExperimentConfigs.read_json(configs_fp)
        fps, _, _, px_per_mm, bins_ls, cbins_ls = AnalyseMixin.get_configs(configs)
        configs_filt = Model_social_distance(**configs.user.analyse.social_distance)  # type: ignore
        bpts = configs.get_ref(configs_filt.bodyparts)
        smoothing_sec = configs.get_ref(configs_filt.smoothing_sec)
        # Calculating more parameters
        smoothing_frames = int(smoothing_sec * fps)

        # Loading in dataframe
        dlc_df = KeypointsMixin.clean_headings(KeypointsMixin.read_feather(dlc_fp))
        # Checking body-centre bodypart exists
        KeypointsMixin.check_bpts_exist(dlc_df, bpts)
        # Getting indivs and bpts list
        indivs, _ = KeypointsMixin.get_headings(dlc_df)

        # Calculating speed of subject for each frame
        analysis_df = AnalyseMixin.init_df(dlc_df.index)
        dlc_df.index = analysis_df.index
        idx = pd.IndexSlice
        # Assumes there are only two individuals
        indiv_a = indivs[0]
        indiv_b = indivs[1]
        # Getting distances between each individual
        idx_a = idx[indiv_b, bpts, "x"]
        dist_x = (dlc_df.loc[:, idx_a] - dlc_df.loc[:, idx_a]).mean(axis=1)  # type: ignore
        idx_b = idx[indiv_a, bpts, "y"]
        dist_y = (dlc_df.loc[:, idx_b] - dlc_df.loc[:, idx_b]).mean(axis=1)  # type: ignore
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
        AggAnalyse.summary_binned_quantitative(
            analysis_df,
            out_dir,
            name,
            fps,
            bins_ls,
            cbins_ls,
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
        f_name = Analyse.freezing.__name__
        out_dir = os.path.join(analysis_dir, f_name)
        # Calculating the deltas (changes in body position) between each frame for the subject
        configs = ExperimentConfigs.read_json(configs_fp)
        fps, _, _, px_per_mm, bins_ls, cbins_ls = AnalyseMixin.get_configs(configs)
        configs_filt = Model_freezing(**configs.user.analyse.freezing)  # type: ignore
        bpts = configs.get_ref(configs_filt.bodyparts)
        thresh_mm = configs.get_ref(configs_filt.thresh_mm)
        smoothing_sec = configs.get_ref(configs_filt.smoothing_sec)
        window_sec = configs.get_ref(configs_filt.window_sec)
        # Calculating more parameters
        thresh_px = thresh_mm / px_per_mm
        smoothing_frames = int(smoothing_sec * fps)
        window_frames = int(np.round(fps * window_sec, 0))

        # Loading in dataframe
        dlc_df = KeypointsMixin.clean_headings(KeypointsMixin.read_feather(dlc_fp))
        # Checking body-centre bodypart exists
        KeypointsMixin.check_bpts_exist(dlc_df, bpts)
        # Getting indivs and bpts list
        indivs, _ = KeypointsMixin.get_headings(dlc_df)

        # Calculating speed of subject for each frame
        analysis_df = AnalyseMixin.init_df(dlc_df.index)
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
            analysis_df[(indiv, f_name)] = temp_df.apply(
                lambda x: pd.Series(np.all(x < thresh_px)), axis=1
            ).astype(np.int8)

            # Getting start, stop, and duration of each freezing behav bout
            freezingbouts_df = BehavDfMixin.vect_2_bouts(
                analysis_df[(indiv, f_name)] == 1
            )
            # For each freezing bout, if there is less than window_frames, tehn
            # it is not actually freezing
            for _, row in freezingbouts_df.iterrows():
                if row["dur"] < window_frames:
                    analysis_df.loc[row["start"] : row["stop"], (indiv, f_name)] = 0
        # Saving analysis_df
        fbf_fp = os.path.join(out_dir, "fbf", f"{name}.feather")
        DFIOMixin.write_feather(analysis_df, fbf_fp)

        # Summarising and binning analysis_df
        AggAnalyse.summary_binned_behavs(
            analysis_df,
            out_dir,
            name,
            fps,
            bins_ls,
            cbins_ls,
        )
        return outcome


class Model_speed(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    smoothing_sec: float | str
    bodyparts: list[str] | str


class Model_social_distance(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    smoothing_sec: float | str
    bodyparts: list[str] | str


class Model_freezing(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    window_sec: float | str
    thresh_mm: float | str
    smoothing_sec: float | str
    bodyparts: list[str] | str


class Model_in_roi(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    roi_name: str
    is_in: bool | str
    thresh_mm: float | str
    roi_corners: list[str] | str
    bodyparts: list[str] | str
