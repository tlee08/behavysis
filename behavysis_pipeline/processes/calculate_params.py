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
from behavysis_core.constants import IndivColumns
from behavysis_core.data_models.experiment_configs import ExperimentConfigs
from behavysis_core.mixins.keypoints_mixin import KeypointsMixin
from pydantic import BaseModel, ConfigDict


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
        - user
            - calculate_params
                - start_frame
                    - window_sec: float
                    - pcutoff: float
        ```
        """
        outcome = ""
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = Model_check_existence(
            **configs.user.calculate_params.start_frame
        )
        bpts = configs.get_ref(configs_filt.bodyparts)
        window_sec = configs.get_ref(configs_filt.window_sec)
        pcutoff = configs.get_ref(configs_filt.pcutoff)
        fps = configs.auto.formatted_vid.fps
        # Asserting that the necessary auto configs are valid
        assert fps is not None, "fps is None. Please calculate fps first."
        # Deriving more parameters
        window_frames = int(np.round(fps * window_sec, 0))
        # Loading dataframe
        dlc_df = KeypointsMixin.clean_headings(KeypointsMixin.read_feather(dlc_fp))
        # Getting likehoods of subject (given bpts) existing in each frame
        df_lhoods = calc_likelihoods(dlc_df, bpts, window_frames)
        # Determining start time. Start frame is the first frame of the rolling window's range
        df_lhoods["exists"] = df_lhoods["rolling"] > pcutoff
        # Getting when subject first and last exists in video
        start_frame = 0
        if np.all(df_lhoods["exists"] == 0):
            # If subject never exists (i.e. no True values in exist column), then raise warning
            outcome += (
                "WARNING: The subject was not detected in any frames - using the first frame."
                + "Please check the video.\n"
            )
        else:
            start_frame = df_lhoods[df_lhoods["exists"]].index[0]
        # Writing to configs
        configs = ExperimentConfigs.read_json(configs_fp)
        configs.auto.start_frame = start_frame
        # configs.auto.start_sec = start_frame / fps
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
        configs_filt = Model_stop_frame(**configs.user.calculate_params.stop_frame)
        dur_sec = configs.get_ref(configs_filt.dur_sec)
        start_frame = configs.auto.start_frame
        fps = configs.auto.formatted_vid.fps
        auto_stop_frame = configs.auto.formatted_vid.total_frames
        # Asserting that the necessary auto configs are valid
        assert (
            start_frame is not None
        ), "start_frame is None. Please calculate start_frame first."
        assert fps is not None, "fps is None. Please calculate fps first."
        assert (
            auto_stop_frame is not None
        ), "total_frames is None. Please calculate total_frames first."
        # Calculating stop_frame
        dur_frames = int(dur_sec * fps)
        stop_frame = start_frame + dur_frames
        # Make a warning if the use-specified dur_sec is larger than the duration of the video.
        if auto_stop_frame is None:
            outcome += (
                "WARNING: The length of the video itself has not been calculated yet."
            )
        elif stop_frame > auto_stop_frame:
            outcome += (
                "WARNING: The user specified dur_sec in the configs file is greater "
                + "than the actual length of the video. Please check to see if this video is "
                + "too short or if the dur_sec value is incorrect.\n"
            )
        # Writing to config
        configs = ExperimentConfigs.read_json(configs_fp)
        configs.auto.stop_frame = stop_frame
        # configs.auto.stop_sec = stop_frame / fps
        configs.write_json(configs_fp)
        return outcome

    @staticmethod
    def exp_dur(dlc_fp: str, configs_fp: str) -> str:
        """
        Calculates the duration in seconds, from the time the specified bodyparts appeared
        to the time they disappeared.
        Appear/disappear is calculated from likelihood.
        """
        outcome = ""
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = Model_check_existence(**configs.user.calculate_params.exp_dur)
        bpts = configs.get_ref(configs_filt.bodyparts)
        window_sec = configs.get_ref(configs_filt.window_sec)
        pcutoff = configs.get_ref(configs_filt.pcutoff)
        fps = configs.auto.formatted_vid.fps
        # Asserting that the necessary auto configs are valid
        assert fps is not None, "fps is None. Please calculate fps first."
        # Deriving more parameters
        window_frames = int(np.round(fps * window_sec, 0))
        # Loading dataframe
        dlc_df = KeypointsMixin.clean_headings(KeypointsMixin.read_feather(dlc_fp))
        # Getting likehoods of subject (given bpts) existing in each frame
        df_lhoods = calc_likelihoods(dlc_df, bpts, window_frames)
        # Determining start time. Start frame is the first frame of the rolling window's range
        df_lhoods["exists"] = df_lhoods["rolling"] > pcutoff
        # Getting when subject first and last exists in video
        exp_dur_frames = 0
        if np.all(df_lhoods["exists"] == 0):
            # If subject never exists (i.e. no True values in exist column), then raise warning
            outcome += (
                "WARNING: The subject was not detected in any frames - using the first frame."
                + "Please check the video.\n"
            )
        else:
            start_frame = df_lhoods[df_lhoods["exists"]].index[0]
            stop_frame = df_lhoods[df_lhoods["exists"]].index[-1]
            exp_dur_frames = stop_frame - start_frame
        # Writing to configs
        configs = ExperimentConfigs.read_json(configs_fp)
        configs.auto.exp_dur_frames = exp_dur_frames
        # configs.auto.exp_dur_secs = exp_dur_frames / fps
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
        - user
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
        configs_filt = Model_px_per_mm(**configs.user.calculate_params.px_per_mm)
        pt_a = configs.get_ref(configs_filt.pt_a)
        pt_b = configs.get_ref(configs_filt.pt_b)
        pcutoff = configs.get_ref(configs_filt.pcutoff)
        dist_mm = configs.get_ref(configs_filt.dist_mm)
        # Loading dataframe
        dlc_df = KeypointsMixin.clean_headings(KeypointsMixin.read_feather(dlc_fp))
        # Imputing missing values with 0 (only really relevant for "likelihood" columns)
        dlc_df = dlc_df.fillna(0)
        # Checking that the two reference points are valid
        KeypointsMixin.check_bpts_exist(dlc_df, [pt_a, pt_b])
        # Getting calibration points (x, y, likelihood) values
        pt_a_df = dlc_df[IndivColumns.SINGLE.value, pt_a]
        pt_b_df = dlc_df[IndivColumns.SINGLE.value, pt_b]
        # Interpolating points which are below a likelihood threshold (linear)
        pt_a_df.loc[pt_a_df["likelihood"] < pcutoff] = np.nan
        pt_a_df = pt_a_df.interpolate(method="linear", axis=0).bfill()
        pt_b_df.loc[pt_b_df["likelihood"] < pcutoff] = np.nan
        pt_b_df = pt_b_df.interpolate(method="linear", axis=0).bfill()
        # Getting distance between calibration points
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


def calc_likelihoods(
    df: pd.DataFrame,
    bpts: list,
    window_frames: int,
):
    """__summary__"""
    # Imputing missing values with 0 (only really relevant for "likelihood" columns)
    df = df.fillna(0)
    # Checking that the two reference points are valid
    KeypointsMixin.check_bpts_exist(df, bpts)
    # Calculating likelihood of subject (given bpts) existing.
    idx = pd.IndexSlice
    df_lhoods = pd.DataFrame(index=df.index)
    df_bpts_lhoods = df.loc[:, idx[:, bpts, "likelihood"]]
    df_lhoods["current"] = df_bpts_lhoods.apply(np.nanmedian, axis=1)
    # Calculating likelihood of subject existing over time window
    df_lhoods["rolling"] = (
        df_lhoods["current"].rolling(window_frames, center=True).agg(np.nanmean)
    )
    # Returning df_lhoods
    return df_lhoods


class Model_stop_frame(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    dur_sec: float | str = 0


class Model_check_existence(BaseModel):
    """__summary__"""

    model_config = ConfigDict(extra="forbid")

    bodyparts: list[str] | str = []
    window_sec: float | str = 0
    pcutoff: float | str = 0


class Model_px_per_mm(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    pt_a: str = "pt_a"
    pt_b: str = "pt_b"
    pcutoff: int | str = 0
    dist_mm: float | str = 0
