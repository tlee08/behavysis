"""
Functions have the following format:

Parameters
----------
keypoints_fp : str
    The DLC dataframe filepath of the experiment to analyse.
dst_dir : str
    The analysis directory path.
configs_fp : str
    the experiment's JSON configs file.

Returns
-------
str
    The outcome of the process.
"""

import logging
import os

import numpy as np
import pandas as pd

from behavysis.df_classes.analysis_agg_df import AnalysisBinnedDf
from behavysis.df_classes.analysis_df import (
    FBF,
    AnalysisDf,
)
from behavysis.df_classes.behav_df import BehavScoredDf
from behavysis.df_classes.keypoints_df import (
    CoordsCols,
    IndivCols,
    KeypointsDf,
)
from behavysis.pydantic_models.experiment_configs import ExperimentConfigs
from behavysis.utils.io_utils import get_name
from behavysis.utils.logging_utils import get_io_obj_content, init_logger_io_obj
from behavysis.utils.misc_utils import get_func_name_in_stack

###################################################################################################
#               ANALYSIS API FUNCS
###################################################################################################


class Analyse:
    @staticmethod
    def in_roi(
        keypoints_fp: str,
        dst_dir: str,
        configs_fp: str,
    ) -> str:
        """
        Determines the frames in which the subject is inside the cage (from average
        of given bodypoints).

        Points are `padding_px` padded (away) from center.
        """
        logger, io_obj = init_logger_io_obj()
        f_name = get_func_name_in_stack()
        name = get_name(keypoints_fp)
        dst_subdir = os.path.join(dst_dir, f_name)
        # Calculating the deltas (changes in body position) between each frame for the subject
        configs = ExperimentConfigs.read_json(configs_fp)
        fps, _, _, px_per_mm, bins_ls, cbins_ls = configs.get_analysis_configs()
        configs_filt_ls = configs.user.analyse.in_roi
        # Loading in dataframe
        keypoints_df = KeypointsDf.clean_headings(KeypointsDf.read(keypoints_fp))
        assert keypoints_df.shape[0] > 0, "No frames in keypoints_df. Please check keypoints file."
        # Getting indivs list
        indivs, _ = KeypointsDf.get_indivs_bpts(keypoints_df)
        # Making analysis_df
        analysis_df_ls = []
        scatter_df_ls = []
        corners_df_ls = []
        roi_names_ls = []
        # For each roi, calculate the in-roi status of the subject
        x = CoordsCols.X.value
        y = CoordsCols.Y.value
        idx = pd.IndexSlice
        for configs_filt in configs_filt_ls:
            # Getting necessary config parameters
            roi_name = configs.get_ref(configs_filt.roi_name)
            is_in = configs.get_ref(configs_filt.is_in)
            bpts = configs.get_ref(configs_filt.bodyparts)
            padding_mm = configs.get_ref(configs_filt.padding_mm)
            roi_corners = configs.get_ref(configs_filt.roi_corners)
            # Calculating more parameters
            padding_px = padding_mm / px_per_mm
            # Checking bodyparts and roi_corners exist
            KeypointsDf.check_bpts_exist(keypoints_df, bpts)
            KeypointsDf.check_bpts_exist(keypoints_df, roi_corners)
            # Getting average corner coordinates. Assumes arena does not move.
            corners_i_df = pd.DataFrame([keypoints_df[(IndivCols.SINGLE.value, pt)].mean() for pt in roi_corners]).drop(
                columns=["likelihood"]
            )
            # Adjusting x-y to have `padding_px` dilation/erosion from the points themselves
            roi_center = corners_i_df.mean()
            for i in corners_i_df.index:
                # Calculating angle from centre to point (going out from centre)
                theta = np.arctan2(
                    corners_i_df.loc[i, y] - roi_center[y],
                    corners_i_df.loc[i, x] - roi_center[x],
                )
                # Getting x, y distances so point is `padding_px` padded (away) from center
                corners_i_df.loc[i, x] = corners_i_df.loc[i, x] + (padding_px * np.cos(theta))
                corners_i_df.loc[i, y] = corners_i_df.loc[i, y] + (padding_px * np.sin(theta))
            # Making the res_df
            analysis_i_df = AnalysisDf.init_df(keypoints_df.index)
            # For each individual, getting the in-roi status
            for indiv in indivs:
                # Getting average body center (x, y) for each individual
                analysis_i_df[(indiv, x)] = keypoints_df.loc[:, idx[indiv, bpts, x]].mean(axis=1).values  # type: ignore
                analysis_i_df[(indiv, y)] = keypoints_df.loc[:, idx[indiv, bpts, y]].mean(axis=1).values  # type: ignore
                # Determining if the indiv body center is in the ROI
                analysis_i_df[(indiv, roi_name)] = analysis_i_df[indiv].apply(
                    lambda pt: pt_in_roi(pt, corners_i_df, logger), axis=1
                )
            # Inverting in_roi status if is_in is False
            if not is_in:
                analysis_i_df.loc[:, idx[:, roi_name]] = ~analysis_i_df.loc[:, idx[:, roi_name]]  # type: ignore
            analysis_df_ls.append(analysis_i_df.loc[:, idx[:, roi_name]].astype(np.int8))  # type: ignore
            scatter_df_ls.append(analysis_i_df)
            corners_df_ls.append(corners_i_df)
            roi_names_ls.append(roi_name)
        # Concatenating all analysis_df_ls and roi_corners_df_ls
        analysis_df = pd.concat(analysis_df_ls, axis=1)
        scatter_df = pd.concat(scatter_df_ls, axis=1)
        corners_df = pd.concat(corners_df_ls, keys=roi_names_ls, names=["roi"]).reset_index(level="roi")
        # Saving analysis_df
        fbf_fp = os.path.join(dst_subdir, FBF, f"{name}.{AnalysisDf.IO}")
        AnalysisDf.write(analysis_df, fbf_fp)
        plot_fp = os.path.join(dst_subdir, "scatter_plot", f"{name}.png")
        AnalysisDf.make_location_scatterplot(scatter_df, corners_df, plot_fp)
        # Summarising and binning analysis_df
        AnalysisBinnedDf.summary_binned_behavs(
            analysis_df,
            dst_subdir,
            name,
            fps,
            bins_ls,
            cbins_ls,
        )
        return get_io_obj_content(io_obj)

    @staticmethod
    def speed(
        keypoints_fp: str,
        dst_dir: str,
        configs_fp: str,
    ) -> str:
        """
        Determines the speed of the subject in each frame.
        """
        logger, io_obj = init_logger_io_obj()
        f_name = get_func_name_in_stack()
        name = get_name(keypoints_fp)
        dst_subdir = os.path.join(dst_dir, f_name)
        # Calculating the deltas (changes in body position) between each frame for the subject
        configs = ExperimentConfigs.read_json(configs_fp)
        fps, _, _, px_per_mm, bins_ls, cbins_ls = configs.get_analysis_configs()
        configs_filt = configs.user.analyse.speed
        bpts = configs.get_ref(configs_filt.bodyparts)
        smoothing_sec = configs.get_ref(configs_filt.smoothing_sec)
        # Calculating more parameters
        smoothing_frames = int(smoothing_sec * fps)

        # Loading in dataframe
        keypoints_df = KeypointsDf.clean_headings(KeypointsDf.read(keypoints_fp))
        assert keypoints_df.shape[0] > 0, "No frames in keypoints_df. Please check keypoints file."
        # Checking body-centre bodypart exists
        KeypointsDf.check_bpts_exist(keypoints_df, bpts)
        # Getting indivs and bpts list
        indivs, _ = KeypointsDf.get_indivs_bpts(keypoints_df)

        # Calculating speed of subject for each frame
        analysis_df = AnalysisDf.init_df(keypoints_df.index)
        # keypoints_df.index = analysis_df.index
        idx = pd.IndexSlice
        for indiv in indivs:
            # Making a rolling window of 3 frames for average body-centre
            # Otherwise jitter contributes to movement
            jitter_frames = 3
            smoothed_xy_df = keypoints_df.rolling(window=jitter_frames, min_periods=1, center=True).agg(np.nanmean)
            # Getting changes in x-y values between frames (deltas)
            delta_x = smoothed_xy_df.loc[:, idx[indiv, bpts, "x"]].mean(axis=1).diff()  # type: ignore
            delta_y = smoothed_xy_df.loc[:, idx[indiv, bpts, "y"]].mean(axis=1).diff()  # type: ignore
            delta = np.array(np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2)))
            # Storing speed (raw and smoothed)
            analysis_df[(indiv, "SpeedMMperSec")] = (delta / px_per_mm) * fps
            analysis_df[(indiv, "SpeedMMperSecSmoothed")] = (
                analysis_df[(indiv, "SpeedMMperSec")]
                .rolling(window=smoothing_frames, min_periods=1, center=True)
                .agg(np.nanmean)
            )
        # Backfilling the analysis_df so no nan's
        analysis_df = analysis_df.bfill()
        # Saving analysis_df
        fbf_fp = os.path.join(dst_subdir, FBF, f"{name}.{AnalysisDf.IO}")
        AnalysisDf.write(analysis_df, fbf_fp)

        # Summarising and binning analysis_df
        AnalysisBinnedDf.summary_binned_quantitative(
            analysis_df,
            dst_subdir,
            name,
            fps,
            bins_ls,
            cbins_ls,
        )
        return get_io_obj_content(io_obj)

    @staticmethod
    def social_distance(
        keypoints_fp: str,
        dst_dir: str,
        configs_fp: str,
    ) -> str:
        """
        Determines the speed of the subject in each frame.
        """
        logger, io_obj = init_logger_io_obj()
        f_name = get_func_name_in_stack()
        name = get_name(keypoints_fp)
        dst_subdir = os.path.join(dst_dir, f_name)
        # Calculating the deltas (changes in body position) between each frame for the subject
        configs = ExperimentConfigs.read_json(configs_fp)
        fps, _, _, px_per_mm, bins_ls, cbins_ls = configs.get_analysis_configs()
        configs_filt = configs.user.analyse.social_distance
        bpts = configs.get_ref(configs_filt.bodyparts)
        smoothing_sec = configs.get_ref(configs_filt.smoothing_sec)
        # Calculating more parameters
        smoothing_frames = int(smoothing_sec * fps)

        # Loading in dataframe
        keypoints_df = KeypointsDf.clean_headings(KeypointsDf.read(keypoints_fp))
        assert keypoints_df.shape[0] > 0, "No frames in keypoints_df. Please check keypoints file."
        # Checking body-centre bodypart exists
        KeypointsDf.check_bpts_exist(keypoints_df, bpts)
        # Getting indivs and bpts list
        indivs, _ = KeypointsDf.get_indivs_bpts(keypoints_df)

        # Calculating speed of subject for each frame
        analysis_df = AnalysisDf.init_df(keypoints_df.index)
        idx = pd.IndexSlice
        # Assumes there are only two individuals
        indiv_a = indivs[0]
        indiv_b = indivs[1]
        # Getting distances between each individual
        idx_a = idx[indiv_b, bpts, "x"]
        dist_x = (keypoints_df.loc[:, idx_a] - keypoints_df.loc[:, idx_a]).mean(axis=1)  # type: ignore
        idx_b = idx[indiv_a, bpts, "y"]
        dist_y = (keypoints_df.loc[:, idx_b] - keypoints_df.loc[:, idx_b]).mean(axis=1)  # type: ignore
        dist = np.array(np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2)))
        # Adding mm distance to saved analysis_df table
        analysis_df[(f"{indiv_a}_{indiv_b}", "DistMM")] = dist / px_per_mm
        analysis_df[(f"{indiv_a}_{indiv_b}", "DistMMSmoothed")] = (
            analysis_df[(f"{indiv_a}_{indiv_b}", "DistMM")]
            .rolling(window=smoothing_frames, min_periods=1, center=True)
            .agg(np.nanmean)
        )
        # Saving analysis_df
        fbf_fp = os.path.join(dst_subdir, FBF, f"{name}.{AnalysisDf.IO}")
        AnalysisDf.write(analysis_df, fbf_fp)

        # Summarising and binning analysis_df
        AnalysisBinnedDf.summary_binned_quantitative(
            analysis_df,
            dst_subdir,
            name,
            fps,
            bins_ls,
            cbins_ls,
        )
        return get_io_obj_content(io_obj)

    @staticmethod
    def freezing(
        keypoints_fp: str,
        dst_dir: str,
        configs_fp: str,
    ) -> str:
        """
        Determines the frames in which the subject is frozen.

        "Frozen" is defined as not moving outside of a radius of `threshold_mm`, and only
        includes bouts that last longer than `window_sec` spent seconds.

        NOTE: method is "greedy" because it looks at a freezing bout from earliest possible frame.
        """
        logger, io_obj = init_logger_io_obj()
        f_name = get_func_name_in_stack()
        name = get_name(keypoints_fp)
        dst_subdir = os.path.join(dst_dir, f_name)
        # Calculating the deltas (changes in body position) between each frame for the subject
        configs = ExperimentConfigs.read_json(configs_fp)
        fps, _, _, px_per_mm, bins_ls, cbins_ls = configs.get_analysis_configs()
        configs_filt = configs.user.analyse.freezing
        bpts = configs.get_ref(configs_filt.bodyparts)
        thresh_mm = configs.get_ref(configs_filt.thresh_mm)
        smoothing_sec = configs.get_ref(configs_filt.smoothing_sec)
        window_sec = configs.get_ref(configs_filt.window_sec)
        # Calculating more parameters
        thresh_px = thresh_mm / px_per_mm
        smoothing_frames = int(smoothing_sec * fps)
        window_frames = int(np.round(fps * window_sec, 0))

        # Loading in dataframe
        keypoints_df = KeypointsDf.clean_headings(KeypointsDf.read(keypoints_fp))
        assert keypoints_df.shape[0] > 0, "No frames in keypoints_df. Please check keypoints file."
        # Checking body-centre bodypart exists
        KeypointsDf.check_bpts_exist(keypoints_df, bpts)
        # Getting indivs and bpts list
        indivs, _ = KeypointsDf.get_indivs_bpts(keypoints_df)

        # Calculating speed of subject for each frame
        analysis_df = AnalysisDf.init_df(keypoints_df.index)
        keypoints_df.index = analysis_df.index
        for indiv in indivs:
            temp_df = pd.DataFrame(index=analysis_df.index)
            # Calculating frame-by-frame delta distances for current bpt
            for bpt in bpts:
                # Getting x and y changes
                delta_x = keypoints_df[(indiv, bpt, "x")].diff()
                delta_y = keypoints_df[(indiv, bpt, "y")].diff()
                # Getting Euclidean distance between frames for bpt
                delta = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))
                # Converting from px to mm
                temp_df[f"{bpt}_dist"] = delta
                # Smoothing
                temp_df[f"{bpt}_dist"] = (
                    temp_df[f"{bpt}_dist"].rolling(window=smoothing_frames, min_periods=1, center=True).agg(np.nanmean)
                )
            # If ALL bodypoints do not leave `thresh_px`
            analysis_df[(indiv, f_name)] = temp_df.apply(lambda x: pd.Series(np.all(x < thresh_px)), axis=1).astype(
                np.int8
            )

            # Getting start, stop, and duration of each freezing behav bout
            freezingbouts_df = BehavScoredDf.vect2bouts_df(analysis_df[(indiv, f_name)] == 1)
            # For each freezing bout, if there is less than window_frames, tehn
            # it is not actually freezing
            for _, row in freezingbouts_df.iterrows():
                if row["dur"] < window_frames:
                    analysis_df.loc[row["start"] : row["stop"], (indiv, f_name)] = 0
        # Saving analysis_df
        fbf_fp = os.path.join(dst_subdir, FBF, f"{name}.{AnalysisDf.IO}")
        AnalysisDf.write(analysis_df, fbf_fp)

        # Summarising and binning analysis_df
        AnalysisBinnedDf.summary_binned_behavs(
            analysis_df,
            dst_subdir,
            name,
            fps,
            bins_ls,
            cbins_ls,
        )
        return get_io_obj_content(io_obj)


def pt_in_roi(pt: pd.Series, corners_df: pd.DataFrame, logger: logging.Logger) -> bool:
    """__summary__"""
    # Counting crossings over edge in region when point is translated to the right
    crossings = 0
    # To loop back to the first point at the end
    first_corner = pd.DataFrame(corners_df.iloc[0]).T
    corners_df = pd.concat((corners_df, first_corner), axis=0, ignore_index=True)
    # Making x and y aliases
    x = CoordsCols.X.value
    y = CoordsCols.Y.value
    # For each edge
    for i in range(corners_df.shape[0] - 1):
        # Getting corner points of edge
        c1 = corners_df.iloc[i]
        c2 = corners_df.iloc[i + 1]
        # Getting whether point-y is between corners-y
        y_between = (c1[y] > pt[y]) != (c2[y] > pt[y])
        # Getting whether point-x is to the left (less than) the intersection of corners-x
        x_left_of = pt[x] < (c2[x] - c1[x]) * (pt[y] - c1[y]) / (c2[y] - c1[y]) + c1[x]
        if y_between and x_left_of:
            crossings += 1
    # Odd number of crossings means point is in region
    return crossings % 2 == 1
