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
from pydantic import BaseModel

from behavysis_core.constants import AnalysisCN, IndivColumns
from behavysis_core.data_models.experiment_configs import ExperimentConfigs
from behavysis_core.mixins.behav_mixin import BehavMixin
from behavysis_core.mixins.df_io_mixin import DFIOMixin
from behavysis_core.mixins.io_mixin import IOMixin
from behavysis_core.mixins.keypoints_mixin import KeypointsMixin

from .analyse_mixin import AggAnalyse, AnalyseMixin

#####################################################################
#               ANALYSIS API FUNCS
#####################################################################


class Analyse:
    """__summary__"""

    @staticmethod
    def thigmotaxis(
        dlc_fp: str,
        analysis_dir: str,
        configs_fp: str,
    ) -> str:
        """
        Determines the frames when the subject is in thigmotaxis.

        Takes DLC data as input and returns the following analysis output:

        - A feather file with the ROI data columns for each video frame (row)
        - A png of the scatterplot of the subject's x-y position in every frame,
        coloured by whether it was in ROI.
        - A png of the bivariate histogram distribution of the subject's x-y position
        for all frames, coloured by whether it was in ROI.
        """
        outcome = ""
        name = IOMixin.get_name(dlc_fp)
        f_name = Analyse.thigmotaxis.__name__
        out_dir = os.path.join(analysis_dir, f_name)
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        fps, _, _, px_per_mm, bins_ls, cbins_ls = AnalyseMixin.get_configs(configs)
        configs_filt = Model_in_roi(**configs.user.analyse.thigmotaxis)
        bpts = configs.get_ref(configs_filt.bodyparts)
        thresh_mm = configs.get_ref(configs_filt.thresh_mm)
        tl = configs.get_ref(configs_filt.roi_top_left)
        tr = configs.get_ref(configs_filt.roi_top_right)
        br = configs.get_ref(configs_filt.roi_bottom_right)
        bl = configs.get_ref(configs_filt.roi_bottom_left)
        # Calculating more parameters
        thresh_px = thresh_mm / px_per_mm

        # Loading in dataframe
        dlc_df = KeypointsMixin.clean_headings(KeypointsMixin.read_feather(dlc_fp))
        # Checking body-centre bodypart exists
        KeypointsMixin.check_bpts_exist(dlc_df, bpts)
        # Getting indivs list
        indivs, _ = KeypointsMixin.get_headings(dlc_df)

        # Getting average corner coordinates. Assumes arena does not move.
        tl = dlc_df[(IndivColumns.SINGLE.value, tl)].mean()
        tr = dlc_df[(IndivColumns.SINGLE.value, tr)].mean()
        br = dlc_df[(IndivColumns.SINGLE.value, br)].mean()
        bl = dlc_df[(IndivColumns.SINGLE.value, bl)].mean()
        # Making roi_df of corners (with the thresh_px buffer)
        roi_df = pd.DataFrame(
            [
                (tl["x"] + thresh_px, tl["y"] + thresh_px),
                (tr["x"] - thresh_px, tr["y"] + thresh_px),
                (br["x"] - thresh_px, br["y"] - thresh_px),
                (bl["x"] + thresh_px, bl["y"] - thresh_px),
            ],
            columns=["x", "y"],
        )
        # Getting the (x, y, in-roi) df
        idx = pd.IndexSlice
        res_df = AnalyseMixin.pt_in_roi_df(dlc_df, roi_df, indivs, bpts)
        # Changing column MultiIndex names
        res_df.columns = res_df.columns.set_levels(["x", "y", f_name], level=1)
        # Setting thigmotaxis as OUTSIDE region (negative)
        res_df.loc[:, idx[:, f_name]] = (res_df.loc[:, idx[:, f_name]] == 0).astype(
            np.int8
        )
        # Getting analysis_df
        analysis_df = res_df.loc[:, idx[:, f_name]]
        # Saving analysis_df
        fbf_fp = os.path.join(out_dir, "fbf", f"{name}.feather")
        DFIOMixin.write_feather(analysis_df, fbf_fp)

        # Generating scatterplot
        plot_fp = os.path.join(out_dir, "scatter_plot", f"{name}.png")
        AnalyseMixin.make_location_scatterplot(res_df, roi_df, plot_fp, f_name)

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

    @staticmethod
    def center_crossing(
        dlc_fp: str,
        analysis_dir: str,
        configs_fp: str,
    ) -> str:
        """
        Determines the frames when the subject is in center.

        Takes DLC data as input and returns the following analysis output:

        - A feather file with the ROI data columns for each video frame (row)
        - A png of the scatterplot of the subject's x-y position in every frame, coloured by whether
        it was in ROI.
        - A png of the bivariate histogram distribution of the subject's x-y position for all
        frames, coloured by whether it was in ROI.
        """
        outcome = ""
        name = IOMixin.get_name(dlc_fp)
        f_name = Analyse.center_crossing.__name__
        out_dir = os.path.join(analysis_dir, f_name)
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        fps, _, _, px_per_mm, bins_ls, cbins_ls = AnalyseMixin.get_configs(configs)
        configs_filt = Model_in_roi(**configs.user.analyse.center_crossing)
        bpts = configs.get_ref(configs_filt.bodyparts)
        thresh_mm = configs.get_ref(configs_filt.thresh_mm)
        tl = configs.get_ref(configs_filt.roi_top_left)
        tr = configs.get_ref(configs_filt.roi_top_right)
        bl = configs.get_ref(configs_filt.roi_bottom_left)
        br = configs.get_ref(configs_filt.roi_bottom_right)
        # Calculating more parameters
        thresh_px = thresh_mm / px_per_mm

        # Loading in dataframe
        dlc_df = KeypointsMixin.clean_headings(KeypointsMixin.read_feather(dlc_fp))
        # Checking body-centre bodypart exists
        KeypointsMixin.check_bpts_exist(dlc_df, bpts)
        # Getting indivs list
        indivs, _ = KeypointsMixin.get_headings(dlc_df)

        # Getting average corner coordinates. Assumes arena does not move.
        tl = dlc_df[(IndivColumns.SINGLE.value, tl)].mean()
        tr = dlc_df[(IndivColumns.SINGLE.value, tr)].mean()
        bl = dlc_df[(IndivColumns.SINGLE.value, bl)].mean()
        br = dlc_df[(IndivColumns.SINGLE.value, br)].mean()
        # Making roi_df of corners (with the thresh_px buffer)
        roi_df = pd.DataFrame(
            [
                (tl["x"] + thresh_px, tl["y"] + thresh_px),
                (tr["x"] - thresh_px, tr["y"] + thresh_px),
                (br["x"] - thresh_px, br["y"] - thresh_px),
                (bl["x"] + thresh_px, bl["y"] - thresh_px),
            ],
            columns=["x", "y"],
        )
        # Getting the (x, y, in-roi) df
        idx = pd.IndexSlice
        res_df = AnalyseMixin.pt_in_roi_df(dlc_df, roi_df, indivs, bpts)
        # Changing column MultiIndex names
        res_df.columns = res_df.columns.set_levels(["x", "y", f_name], level=1)
        # Getting analysis_df
        analysis_df = res_df.loc[:, idx[:, f_name]]
        # Saving analysis_df
        fbf_fp = os.path.join(out_dir, "fbf", f"{name}.feather")
        DFIOMixin.write_feather(analysis_df, fbf_fp)

        # Generating scatterplot
        plot_fp = os.path.join(out_dir, "scatter_plot", f"{name}.png")
        AnalyseMixin.make_location_scatterplot(res_df, roi_df, plot_fp, f_name)

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
        configs_filt_ls = list(configs.user.analyse.in_roi)
        # Loading in dataframe
        dlc_df = KeypointsMixin.clean_headings(KeypointsMixin.read_feather(dlc_fp))
        # Getting indivs list
        indivs, _ = KeypointsMixin.get_headings(dlc_df)
        # Making analysis_df
        analysis_df_ls = []
        # For each roi, calculate the in-roi status of the subject
        idx = pd.IndexSlice
        for configs_filt in configs_filt_ls:
            # Getting necessary config parameters
            configs_filt = Model_in_roi(**configs_filt)
            roi_name = configs.get_ref(configs_filt.roi_name)
            bpts = configs.get_ref(configs_filt.bodyparts)
            thresh_mm = configs.get_ref(configs_filt.thresh_mm)
            roi_corners = configs.get_ref(configs_filt.roi_corners)
            # Calculating more parameters
            thresh_px = thresh_mm / px_per_mm
            # Checking bodyparts and roi_corners exist
            KeypointsMixin.check_bpts_exist(dlc_df, bpts)
            KeypointsMixin.check_bpts_exist(dlc_df, roi_corners)
            # Getting average corner coordinates. Assumes arena does not move.
            roi_corners_df = pd.DataFrame(
                [dlc_df[(IndivColumns.SINGLE.value, pt)].mean() for pt in roi_corners]
            ).drop(columns=["likelihood"])
            # Adjusting x-y to have a distance dilation/erosion from the points themselves
            roi_center = roi_corners_df.mean()
            for i in roi_corners_df.index:
                # Calculating angle from point to centre
                theta = np.arctan2(
                    roi_corners_df.loc[i, "y"] - roi_center["y"],
                    roi_corners_df.loc[i, "x"] - roi_center["x"],
                )
                # Getting x, y distances so point is `thresh_px` closer to center
                roi_corners_df.loc[i, "x"] = roi_corners_df.loc[i, "x"] - (
                    thresh_px * np.cos(theta)
                )
                roi_corners_df.loc[i, "y"] = roi_corners_df.loc[i, "y"] - (
                    thresh_px * np.sin(theta)
                )
            # Getting the (x, y, in-roi) df
            res_df = AnalyseMixin.pt_in_roi_df(dlc_df, roi_corners_df, indivs, bpts)
            # Changing column MultiIndex names
            res_df.columns = res_df.columns.set_levels(
                ["x", "y", f"in_roi_{roi_name}"], level=AnalysisCN.MEASURES.value
            )
            # Saving to analysis_df list
            analysis_df_ls.append(res_df.loc[:, idx[:, f"in_roi_{roi_name}"]])
        # Concatenating all analysis_df_ls
        analysis_df = pd.concat(analysis_df_ls, axis=1)
        # Saving analysis_df
        fbf_fp = os.path.join(out_dir, "fbf", f"{name}.feather")
        DFIOMixin.write_feather(analysis_df, fbf_fp)
        # Generating scatterplot
        # First getting scatter_in_roi columns
        scatter_df_ls = [
            pd.DataFrame(
                analysis_df[i].apply(lambda x: "-".join(x.index[x == 1]), axis=1),
                columns=pd.MultiIndex.from_tuples(
                    [(i, "roi")], names=DFIOMixin.enum_to_list(AnalysisCN)
                ),
            )
            for i in indivs
        ]
        # Concatenating, and including most recent x, y columns
        # TODO: any way to include all different "x", "y" to use?
        scatter_df = pd.concat(
            [res_df.loc[:, idx[:, ["x", "y"]]], *scatter_df_ls], axis=1
        )
        scatter_df.columns = pd.MultiIndex.from_tuples(
            scatter_df.columns, names=DFIOMixin.enum_to_list(AnalysisCN)
        )
        print(scatter_df)
        # Making and saving scatterplot
        plot_fp = os.path.join(out_dir, "scatter_plot", f"{name}.png")
        AnalyseMixin.make_location_scatterplot(
            scatter_df, roi_corners_df, plot_fp, "roi"
        )
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
    def in_roi_el(
        name: str,
        dlc_df: pd.DataFrame,
        configs: ExperimentConfigs,
        configs_filt: Model_in_roi,
        out_dir: str,
    ):
        # Getting necessary config parameters
        fps, _, _, px_per_mm, bins_ls, cbins_ls = AnalyseMixin.get_configs(configs)
        roi_name = configs.get_ref(configs_filt.roi_name)
        bpts = configs.get_ref(configs_filt.bodyparts)
        thresh_mm = configs.get_ref(configs_filt.thresh_mm)
        roi_corners = configs.get_ref(configs_filt.roi_corners)
        # Calculating more parameters
        thresh_px = thresh_mm / px_per_mm
        f_name = f"in_roi_{roi_name}"
        # Checking bodyparts and roi_corners exist
        KeypointsMixin.check_bpts_exist(dlc_df, bpts)
        KeypointsMixin.check_bpts_exist(dlc_df, roi_corners)
        # Getting indivs list
        indivs, _ = KeypointsMixin.get_headings(dlc_df)
        # Getting average corner coordinates. Assumes arena does not move.
        roi_corners_df = pd.DataFrame(
            [dlc_df[(IndivColumns.SINGLE.value, pt)].mean() for pt in roi_corners]
        ).drop(columns=["likelihood"])
        # Adjusting x-y to have a distance dilation/erosion from the points themselves
        roi_center = roi_corners_df.mean()
        for i in roi_corners_df.index:
            # Calculating angle from point to centre
            theta = np.arctan2(
                roi_corners_df.loc[i, "y"] - roi_center["y"],
                roi_corners_df.loc[i, "x"] - roi_center["x"],
            )
            # Getting x, y distances so point is `thresh_px` closer to center
            roi_corners_df.loc[i, "x"] = roi_corners_df.loc[i, "x"] - (
                thresh_px * np.cos(theta)
            )
            roi_corners_df.loc[i, "y"] = roi_corners_df.loc[i, "y"] - (
                thresh_px * np.sin(theta)
            )
        # Getting the (x, y, in-roi) df
        res_df = AnalyseMixin.pt_in_roi_df(dlc_df, roi_corners_df, indivs, bpts)
        # Changing column MultiIndex names
        res_df.columns = res_df.columns.set_levels(
            ["x", "y", f_name], level=AnalysisCN.MEASURES.value
        )

        # Getting analysis_df
        idx = pd.IndexSlice
        analysis_df = res_df.loc[:, idx[:, f_name]]
        # Saving analysis_df
        fbf_fp = os.path.join(out_dir, "fbf", f"{name}.feather")
        DFIOMixin.write_feather(analysis_df, fbf_fp)
        # Generating scatterplot
        plot_fp = os.path.join(out_dir, "scatter_plot", f"{name}.png")
        AnalyseMixin.make_location_scatterplot(res_df, roi_corners_df, plot_fp, f_name)
        # Summarising and binning analysis_df
        AggAnalyse.summary_binned_behavs(
            analysis_df,
            out_dir,
            name,
            fps,
            bins_ls,
            cbins_ls,
        )

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
        configs_filt = Model_speed(**configs.user.analyse.speed)
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
            delta_x = smoothed_xy_df.loc[:, idx[indiv, bpts, "x"]].mean(axis=1).diff()
            delta_y = smoothed_xy_df.loc[:, idx[indiv, bpts, "y"]].mean(axis=1).diff()
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
        configs_filt = Model_social_distance(**configs.user.analyse.social_distance)
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
        dist_x = (dlc_df.loc[:, idx_a] - dlc_df.loc[:, idx_a]).mean(axis=1)
        idx_b = idx[indiv_a, bpts, "y"]
        dist_y = (dlc_df.loc[:, idx_b] - dlc_df.loc[:, idx_b]).mean(axis=1)
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
        configs_filt = Model_freezing(**configs.user.analyse.freezing)
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
            freezingbouts_df = BehavMixin.vect_2_bouts(
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

    smoothing_sec: float | str
    bodyparts: list[str] | str


class Model_social_distance(BaseModel):
    """_summary_"""

    smoothing_sec: float | str
    bodyparts: list[str] | str


class Model_freezing(BaseModel):
    """_summary_"""

    window_sec: float | str
    thresh_mm: float | str
    smoothing_sec: float | str
    bodyparts: list[str] | str


class Model_in_roi(BaseModel):
    """_summary_"""

    roi_name: str
    thresh_mm: float | str
    roi_corners: list[str] | str
    bodyparts: list[str] | str
