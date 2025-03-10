"""
Functions have the following format:

Parameters
----------
keypoints_fp : str
    The experiment's keypoints file.
configs_fp : str
    The experiment's JSON configs file.

Returns
-------
str
    The outcome of the process.
"""

import logging

import numpy as np
import pandas as pd

from behavysis.df_classes.keypoints_df import (
    CoordsCols,
    IndivCols,
    KeypointsDf,
)
from behavysis.pydantic_models.experiment_configs import ExperimentConfigs
from behavysis.utils.io_utils import get_name
from behavysis.utils.logging_utils import get_io_obj_content, init_logger_io_obj


class CalculateParams:
    @staticmethod
    def start_frame_from_likelihood(
        keypoints_fp: str,
        configs_fp: str,
    ) -> str:
        """
        Determines the starting frame of the experiment based on
        when the subject "likely" entered the frame of view.

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
                    - bodyparts: list[str]
                    - window_sec: float
                    - pcutoff: float
        ```
        """
        logger, io_obj = init_logger_io_obj()
        start_frame, stop_frame = calc_exists_from_likelihood(keypoints_fp, configs_fp, logger)
        # Writing to configs
        configs = ExperimentConfigs.read_json(configs_fp)
        configs.auto.start_frame = start_frame
        configs.write_json(configs_fp)
        return get_io_obj_content(io_obj)

    @staticmethod
    def start_frame_from_csv(keypoints_fp: str, configs_fp: str) -> str:
        """
        Reads the start time of the experiment from a given CSV file
        (filepath specified in config file).

        Expects value to be in seconds (so will convert to frames).
        Also expects the csv_fp to be a csv file,
        where the first column is the name of the video and the second column
        is the start time.

        Notes
        -----
        The config file must contain the following parameters:
        ```
        - user
            - calculate_params
                - start_frame_from_csv
                    - csv_fp: str
                    - name: None | str
        ```
        """
        logger, io_obj = init_logger_io_obj()
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = configs.user.calculate_params.start_frame_from_csv
        fps = configs.auto.formatted_vid.fps
        csv_fp = configs.get_ref(configs_filt.csv_fp)
        name = configs.get_ref(configs_filt.name)
        assert fps != -1, "fps not yet set. Please calculate fps first with `proj.get_vid_metadata`."
        # Using the name of the video as the name of the experiment if not specified
        if name is None:
            name = get_name(keypoints_fp)
        # Reading csv_fp
        start_times_df = pd.read_csv(csv_fp, index_col=0)
        start_times_df.index = start_times_df.index.astype(str)
        assert name in start_times_df.index.values, (
            f"{name} not in {csv_fp}.\n"
            "Update the `name` parameter in the configs file or check the start_frames csv file."
        )
        # Getting start time in seconds
        start_sec = start_times_df.loc[name][0]
        # Converting to start frame
        start_frame = int(np.round(start_sec * fps, 0))
        # Writing to configs
        configs = ExperimentConfigs.read_json(configs_fp)
        configs.auto.start_frame = start_frame
        configs.write_json(configs_fp)
        return get_io_obj_content(io_obj)

    @staticmethod
    def stop_frame_from_likelihood(keypoints_fp: str, configs_fp: str) -> str:
        """
        Determines the starting frame of the experiment based on
        when the subject "likely" entered the frame of view.

        This is done by looking at a sliding window of time. If the median likelihood of the subject
        existing in each frame across the sliding window is greater than the defined pcutoff, then
        the determine this as the start time.

        """
        logger, io_obj = init_logger_io_obj()
        start_frame, stop_frame = calc_exists_from_likelihood(keypoints_fp, configs_fp, logger)
        # Writing to configs
        configs = ExperimentConfigs.read_json(configs_fp)
        configs.auto.stop_frame = stop_frame
        configs.write_json(configs_fp)
        return get_io_obj_content(io_obj)

    @staticmethod
    def stop_frame_from_dur(keypoints_fp: str, configs_fp: str) -> str:
        """
        Calculates the end time according to the following equation:

        ```
        stop_frame = start_frame + experiment_duration
        ```

        Notes
        -----
        The config file must contain the following parameters:
        ```
        - user
            - calculate_params
                - stop_frame_from_dur
                    - dur_sec: float
        ```
        """
        logger, io_obj = init_logger_io_obj()
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = configs.user.calculate_params.stop_frame_from_dur
        dur_sec = configs.get_ref(configs_filt.dur_sec)
        start_frame = configs.auto.start_frame
        fps = configs.auto.formatted_vid.fps
        total_frames = configs.auto.formatted_vid.total_frames
        assert start_frame != -1, "start_frame is None. Please calculate start_frame first."
        assert fps != -1, "fps not yet set. Please calculate fps first with `proj.get_vid_metadata`."
        # Calculating stop_frame
        dur_frames = int(dur_sec * fps)
        stop_frame = start_frame + dur_frames
        # Make a warning if the use-specified dur_sec is larger than the duration of the video.
        if total_frames is None:
            logger.warning("The length of the video itself has not been calculated yet.")
        elif stop_frame > total_frames:
            logger.warning(
                "The user specified dur_sec in the configs file is greater "
                "than the actual length of the video. Please check to see if this video is "
                "too short or if the dur_sec value is incorrect."
            )
        # Writing to config
        configs = ExperimentConfigs.read_json(configs_fp)
        configs.auto.stop_frame = stop_frame
        configs.write_json(configs_fp)
        return get_io_obj_content(io_obj)

    @staticmethod
    def dur_frames_from_likelihood(keypoints_fp: str, configs_fp: str) -> str:
        """
        Calculates the duration in seconds, from the time the specified bodyparts appeared
        to the time they disappeared.
        Appear/disappear is calculated from likelihood.
        """
        logger, io_obj = init_logger_io_obj()
        start_frame, stop_frame = calc_exists_from_likelihood(keypoints_fp, configs_fp, logger)
        # Writing to configs
        configs = ExperimentConfigs.read_json(configs_fp)
        configs.auto.dur_frames = stop_frame - start_frame
        configs.write_json(configs_fp)
        return get_io_obj_content(io_obj)

    @staticmethod
    def px_per_mm(keypoints_fp: str, configs_fp: str) -> str:
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
        logger, io_obj = init_logger_io_obj()
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = configs.user.calculate_params.px_per_mm
        pt_a = configs.get_ref(configs_filt.pt_a)
        pt_b = configs.get_ref(configs_filt.pt_b)
        pcutoff = configs.get_ref(configs_filt.pcutoff)
        dist_mm = configs.get_ref(configs_filt.dist_mm)
        # Loading dataframe
        keypoints_df = KeypointsDf.clean_headings(KeypointsDf.read(keypoints_fp))
        # Imputing missing values with 0 (only really relevant for `likelihood` columns)
        keypoints_df = keypoints_df.fillna(0)
        # Checking that the two reference points are valid
        KeypointsDf.check_bpts_exist(keypoints_df, [pt_a, pt_b])
        # Getting calibration points (x, y, likelihood) values
        pt_a_df = keypoints_df[IndivCols.SINGLE.value, pt_a]
        pt_b_df = keypoints_df[IndivCols.SINGLE.value, pt_b]
        for pt_df, pt in ([pt_a_df, pt_a], [pt_b_df, pt_b]):
            assert np.any(pt_df[CoordsCols.LIKELIHOOD.value] > pcutoff), (
                f'No points for "{pt}" are above the pcutoff of {pcutoff}.\n'
                "Consider lowering the pcutoff in the configs file.\n"
                f'The highest likelihood value in "{pt}" is {np.nanmax(pt_df[CoordsCols.LIKELIHOOD.value])}.'
            )
        # Interpolating points which are below a likelihood threshold (linear)
        pt_a_df.loc[pt_a_df[CoordsCols.LIKELIHOOD.value] < pcutoff] = np.nan
        pt_a_df = pt_a_df.interpolate(method="linear", axis=0).bfill().ffill()
        pt_b_df.loc[pt_b_df[CoordsCols.LIKELIHOOD.value] < pcutoff] = np.nan
        pt_b_df = pt_b_df.interpolate(method="linear", axis=0).bfill().ffill()
        # Getting distance between calibration points
        # TODO: use variable names for x and y
        dist_px = np.nanmean(np.sqrt(np.square(pt_a_df["x"] - pt_b_df["x"]) + np.square(pt_a_df["y"] - pt_b_df["y"])))
        # Finding pixels per mm conversion, using the given arena width and height as calibration
        px_per_mm = dist_px / dist_mm
        # Saving to configs file
        configs = ExperimentConfigs.read_json(configs_fp)
        configs.auto.px_per_mm = px_per_mm
        configs.write_json(configs_fp)
        return get_io_obj_content(io_obj)


def calc_exists_from_likelihood(keypoints_fp: str, configs_fp: str, logger: logging.Logger) -> tuple[int, int]:
    """
    Determines the start and stop frames of the experiment based on
    when the subject "likely" entered and exited the frame of view.

    This is done by looking at a sliding window of time. If the median likelihood of the subject
    existing in each frame across the sliding window is greater than the defined pcutoff, then
    the determine this as the start time.

    Notes
    -----
    The config file must contain the following parameters:
    ```
    - user
        - calculate_params
            - from_likelihood
                - bodyparts: list[str]
                - window_sec: float
                - pcutoff: float
    ```
    """
    # Getting necessary config parameters
    configs = ExperimentConfigs.read_json(configs_fp)
    configs_filt = configs.user.calculate_params.from_likelihood
    bpts = configs.get_ref(configs_filt.bodyparts)
    window_sec = configs.get_ref(configs_filt.window_sec)
    pcutoff = configs.get_ref(configs_filt.pcutoff)
    fps = configs.auto.formatted_vid.fps
    assert fps != -1, "fps not yet set. Please calculate fps first with `proj.get_vid_metadata`."
    # Deriving more parameters
    window_frames = int(np.round(fps * window_sec, 0))
    # Loading dataframe
    keypoints_df = KeypointsDf.clean_headings(KeypointsDf.read(keypoints_fp))
    # Getting likehoods of subject (given bpts) existing in each frame
    KeypointsDf.check_bpts_exist(keypoints_df, bpts)
    idx = pd.IndexSlice
    lhood_name = CoordsCols.LIKELIHOOD.value
    lhood_df = pd.DataFrame(index=keypoints_df.index)
    indivs, _ = KeypointsDf.get_indivs_bpts(keypoints_df)
    for indiv in indivs:
        # Calculating likelihood of subject existing at each frame from median
        lhood_df[(indiv, "current")] = keypoints_df.loc[:, idx[indiv, bpts, lhood_name]].apply(np.nanmedian, axis=1)  # type: ignore
        # Calculating likelihood of subject existing over time window
        lhood_df[(indiv, "rolling")] = lhood_df[(indiv, "current")].rolling(window_frames, center=True).agg(np.nanmean)
    lhood_df.columns = pd.MultiIndex.from_tuples(lhood_df.columns)
    # Getting bool of frames where ALL indivs exist
    idx = pd.IndexSlice
    exists_vect = (lhood_df.loc[:, idx[:, "rolling"]] > pcutoff).all(axis=1)  # type: ignore
    assert np.any(exists_vect), "The subject was not detected in any frames. Please also check the video."
    # Getting when subject first and last exists in video
    start_frame = lhood_df[exists_vect].index[0]
    stop_frame = lhood_df[exists_vect].index[-1]
    return start_frame, stop_frame
