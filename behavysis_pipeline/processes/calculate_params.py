"""
Functions have the following format:

Parameters
----------
dlc_fp : str
    The experiment's dlc file.
configs_fp : str
    The experiment's JSON configs file.

Returns
-------
str
    The outcome of the process.
"""

import numpy as np
import pandas as pd
from behavysis_core.mixins.df_io_mixin import DFIOMixin
from behavysis_core.mixins.keypoints_mixin import KeypointsMixin
from behavysis_core.utils.constants import SINGLE_COL

from behavysis_pipeline.pipeline.experiment_configs import ExperimentConfigs


class CalculateParams:
    """__summary__"""

    @staticmethod
    def start_frame(
        dlc_fp: str,
        configs_fp: str,
    ) -> str:
        """
        Determine the starting frame of the experiment based on when the subject "likely" entered
        the footage.

        This is done by looking at a sliding window of time. If the median likelihood of the subject
        existing in each frame across the sliding window is greater than the defined pcutoff, then
        the determine this as the start time.

        Notes
        -----
        The config file must contain the following parameters:
        ```
        - (user, auto)
            - calculate_params
                - start_frame
                    - window_sec: float
                    - pcutoff: float
        ```
        """
        outcome = ""
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = configs.user.calculate_params.start_frame
        bpts = configs_filt.bodyparts
        window_sec = configs_filt.window_sec
        pcutoff = configs_filt.pcutoff
        fps = configs.auto.formatted_vid.fps
        # Deriving more parameters
        window_frames = int(np.round(fps * window_sec, 0))  # for rounding
        # Loading dataframe
        dlc_df = DFIOMixin.read_feather(dlc_fp)
        # Imputing missing values with 0 (only really relevant for "likelihood" columns)
        dlc_df = dlc_df.fillna(0)
        # Checking that the two reference points are valid
        KeypointsMixin.check_bpts_exist(dlc_df, bpts)
        # Calculating likelihood of subject existing.
        idx = pd.IndexSlice
        df_lhoods = pd.DataFrame(index=dlc_df.index)
        df_lhoods["current"] = dlc_df.loc[:, idx[:, :, bpts, "likelihood"]].apply(
            np.nanmean, axis=1
        )
        # Calculating likelihood of subject existing over time window
        df_lhoods["rolling"] = (
            df_lhoods["current"].rolling(window_frames).agg(np.nanmedian)
        )
        # Determining start time. Start frame is the first frame of the rolling window's range
        try:
            start_frame = df_lhoods[df_lhoods["rolling"] > pcutoff].index[0] - (
                window_frames - 1
            )
        except Exception:
            outcome += (
                "WARNING: The subject was not detected in any frames - using the first frame."
                + "Please check the video.\n"
            )
            start_frame = 0
        configs = ExperimentConfigs.read_json(configs_fp)
        configs.auto.start_frame = start_frame
        configs.write_json(configs_fp)
        return outcome

    @staticmethod
    def stop_frame(dlc_fp: str, configs_fp: str) -> str:
        """
        Calculates the end time according to the following equation:

        ```
        stop_frame = start_frame + experiment_duration
        ```

        Notes
        -----
        The config file must contain the following parameters:
        ```
        TODO
        ```
        """
        outcome = ""
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = configs.user.calculate_params.stop_frame
        dur_sec = configs_filt.dur_sec
        start_frame = configs.auto.start_frame
        fps = configs.auto.formatted_vid.fps
        auto_stop_frame = configs.auto.formatted_vid.total_frames
        # Calculating stop_frame
        dur_frames = int(dur_sec * fps)
        stop_frame = start_frame + dur_frames
        # Make a warning if the use-specified dur_sec is larger than the duration of the video.
        if stop_frame > auto_stop_frame:
            outcome += (
                "WARNING: The user specified dur_sec in the configs file is greater "
                + "than the actual length of the video. Please check to see if this video is "
                + "too short or if the dur_sec value is incorrect.\n"
            )
        configs = ExperimentConfigs.read_json(configs_fp)
        configs.auto.stop_frame = stop_frame
        configs.write_json(configs_fp)
        return outcome

    @staticmethod
    def px_per_mm(dlc_fp: str, configs_fp: str) -> str:
        """
        Calculates the pixels per mm conversion for the video.

        This is done by averaging the (x, y) coordinates of each corner,
        finding the average x difference for the widths in pixels and y distance
        for the heights in pixels,
        dividing these pixel distances by their respective mm distances
        (from the *config.json file),
        and taking the average of these width and height conversions to estimate
        the px to mm
        conversion.

        Notes
        -----
        The config file must contain the following parameters:
        ```
        - (user, auto)
            - calculate_params
                - px_per_mm
                    - point_a: str
                    - point_b: str
                    - dist_mm: float
        ```
        """
        outcome = ""
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = configs.user.calculate_params.px_per_mm
        pt_a = configs_filt.pt_a
        pt_b = configs_filt.pt_b
        pcutoff = configs_filt.pcutoff
        dist_mm = configs_filt.dist_mm
        # Loading dataframe
        dlc_df = KeypointsMixin.clean_headings(DFIOMixin.read_feather(dlc_fp))
        # Imputing missing values with 0 (only really relevant for "likelihood" columns)
        dlc_df = dlc_df.fillna(0)
        # Checking that the two reference points are valid
        KeypointsMixin.check_bpts_exist(dlc_df, [pt_a, pt_b])
        # Finding the arena height and width in pixels
        pt_a_df = dlc_df[SINGLE_COL, pt_a]
        pt_b_df = dlc_df[SINGLE_COL, pt_b]
        # Interpolating points
        pt_a_df[pt_a_df["likelihood"] < pcutoff] = np.nan
        pt_a_df = pt_a_df.interpolate(method="linear", axis=0).bfill()
        pt_b_df[pt_b_df["likelihood"] < pcutoff] = np.nan
        pt_b_df = pt_a_df.interpolate(method="linear", axis=0).bfill()
        # Getting distances
        dist_px = np.nanmean(
            np.sqrt(
                np.square(pt_a_df["x"] - pt_b_df["x"])
                + np.square(pt_a_df["y"] - pt_b_df["y"])
            )
        )
        # Finding pixels per mm conversion, using the given arena width and height as calibration
        px_per_mm = dist_px / dist_mm
        # Saving to configs file
        configs = ExperimentConfigs.read_json(configs_fp)
        configs.auto.px_per_mm = px_per_mm
        configs.write_json(configs_fp)
        return outcome
