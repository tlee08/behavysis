"""Functions have the following format."""

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from loguru import logger

from behavysis.constants import LIKELIHOOD, SINGLE
from behavysis.df_classes import KeypointsDf
from behavysis.models import ExperimentConfig


class CalculateParamsFunc(Protocol):
    """Protocol for calculate_params functions."""

    def __call__(
        self,
        keypoints_fp: Path,
        config_fp: Path,
    ) -> None:
        """Protocol for calculate_params functions."""
        ...


def start_frame_from_likelihood(
    keypoints_fp: Path,
    config_fp: Path,
) -> None:
    """Determines start frame based on when subject "likely" entered the frame.

    This is done by looking at a sliding window of time.
    If the median likelihood of the subject
    existing in each frame across the sliding window is
    greater than the defined pcutoff, then
    the determine this as the start time.

    Notes:
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
    start_frame, _stop_frame = _calc_exists_from_likelihood(keypoints_fp, config_fp)
    # Writing to config
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config.auto.start_frame = start_frame
    config_fp.write_text(config.model_dump_json(indent=2))


def start_frame_from_csv(keypoints_fp: Path, config_fp: Path) -> None:
    """Determines start frame from timestamps in csv.

    Expects value to be in seconds (so will convert to frames).
    Also expects the csv_fp to be a csv file,
    where the first column is the name of the video and the second column
    is the start time.
    Also expect a header row, but it doesn't matter what the header names are.

    Notes:
    -----
    The config file must contain the following parameters:
    ```
    - user
        - calculate_params
            - start_frame_from_csv
                - csv_fp: Path
                - name: None | str
    ```
    """
    # Getting necessary config parameters
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config_filt = config.user.calculate_params.start_frame_from_csv
    fps = config.auto.formatted_vid.fps
    csv_fp = config.get_ref(config_filt.csv_fp)
    name = config.get_ref(config_filt.name)
    assert fps != -1, (
        "fps not yet set. Please calculate fps first with `proj.get_vid_metadata`."
    )
    # Using the name of the video as the name of the experiment if not specified
    if name is None:
        name = keypoints_fp.stem
    # Reading csv_fp
    start_times_df = pd.read_csv(csv_fp, index_col=0)
    start_times_df.index = start_times_df.index.astype(str)
    assert name in start_times_df.index.to_numpy(), (
        f"{name} not in {csv_fp}.\n"
        "Update `name` parameter in config file or check the start_frames csv file."
    )
    # Getting start time in seconds
    start_sec = start_times_df.loc[name][0]
    # Converting to start frame
    start_frame = int(np.round(start_sec * fps, 0))
    # Writing to config
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config.auto.start_frame = start_frame
    config_fp.write_text(config.model_dump_json(indent=2))


def stop_frame_from_likelihood(keypoints_fp: Path, config_fp: Path) -> None:
    """Determines stop frame based on when subject "likely" entered the frame.

    This is done by looking at a sliding window of time.
    If the median likelihood of the subject
    existing in each frame across the sliding window
    is greater than the defined pcutoff, then
    the determine this as the start time.
    """
    _start_frame, stop_frame = _calc_exists_from_likelihood(keypoints_fp, config_fp)
    # Writing to config
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config.auto.stop_frame = stop_frame
    config_fp.write_text(config.model_dump_json(indent=2))


def stop_frame_from_dur(
    keypoints_fp: Path,  # noqa: ARG001
    config_fp: Path,
) -> None:
    """Calculates the end time according to the following equation.

    ```
    stop_frame = start_frame + experiment_duration
    ```

    Notes:
    -----
    The config file must contain the following parameters:
    ```
    - user
        - calculate_params
            - stop_frame_from_dur
                - dur_sec: float
    ```
    """
    # Getting necessary config parameters
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config_filt = config.user.calculate_params.stop_frame_from_dur
    dur_sec = config.get_ref(config_filt.dur_sec)
    start_frame = config.auto.start_frame
    fps = config.auto.formatted_vid.fps
    total_frames = config.auto.formatted_vid.total_frames
    assert start_frame != -1, "start_frame is None. Please calculate start_frame first."
    assert fps != -1, (
        "fps not yet set. Please calculate fps first with `proj.get_vid_metadata`."
    )
    # Calculating stop_frame
    dur_frames = int(dur_sec * fps)
    stop_frame = start_frame + dur_frames
    # Make warning if use-specified dur_sec is larger than the video dur.
    if total_frames is None:
        logger.warning("The length of the video itself has not been calculated yet.")
    elif stop_frame > total_frames:
        logger.warning(
            "The user specified dur_sec in the config file is greater "
            "than the actual length of the video. Please check to see if this video is "
            "too short or if the dur_sec value is incorrect.",
        )
    # Writing to config
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config.auto.stop_frame = stop_frame
    config_fp.write_text(config.model_dump_json(indent=2))


def dur_frames_from_likelihood(keypoints_fp: Path, config_fp: Path) -> None:
    """Determines duration in seconds, from subject first to last seen in vid.

    Appear/disappear is calculated from likelihood.
    """
    start_frame, stop_frame = _calc_exists_from_likelihood(keypoints_fp, config_fp)
    # Writing to config
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config.auto.dur_frames = stop_frame - start_frame
    config_fp.write_text(config.model_dump_json(indent=2))


def px_per_mm(keypoints_fp: Path, config_fp: Path) -> None:
    """Calculates the pixels per mm conversion for the video.

    This is done by averaging the (x, y) coordinates of each corner,
    finding the average x difference for the widths in pixels and y distance
    for the heights in pixels,
    dividing these pixel distances by their respective mm distances
    (from the *config.json file),
    and taking the average of these width and height conversions to estimate
    the px to mm
    conversion.

    Notes:
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
    # Getting necessary config parameters
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config_filt = config.user.calculate_params.px_per_mm
    pt_a = config.get_ref(config_filt.pt_a)
    pt_b = config.get_ref(config_filt.pt_b)
    pcutoff = config.get_ref(config_filt.pcutoff)
    dist_mm = config.get_ref(config_filt.dist_mm)
    # Loading dataframe
    keypoints_df = KeypointsDf.clean_headings(KeypointsDf.read(keypoints_fp))
    # Imputing missing values with 0 (only really relevant for `likelihood` columns)
    keypoints_df = keypoints_df.fillna(0)
    # Checking that the two reference points are valid
    KeypointsDf.check_bpts_exist(keypoints_df, [pt_a, pt_b])
    # Getting calibration points (x, y, likelihood) values
    pt_a_df = keypoints_df[SINGLE, pt_a]
    pt_b_df = keypoints_df[SINGLE, pt_b]
    for pt_df, pt in ([pt_a_df, pt_a], [pt_b_df, pt_b]):
        assert np.any(pt_df[LIKELIHOOD] > pcutoff), (
            f'No points for "{pt}" are above the pcutoff of {pcutoff}.\n'
            "Consider lowering the pcutoff in the config file.\n"
            f'The highest likelihood value in "{pt}" is '
            f"{np.nanmax(pt_df[LIKELIHOOD])}."
        )
    # Interpolating points which are below a likelihood threshold (linear)
    pt_a_df.loc[pt_a_df[LIKELIHOOD] < pcutoff] = np.nan
    pt_a_df = pt_a_df.interpolate(method="linear", axis=0).bfill().ffill()
    pt_b_df.loc[pt_b_df[LIKELIHOOD] < pcutoff] = np.nan
    pt_b_df = pt_b_df.interpolate(method="linear", axis=0).bfill().ffill()
    # Getting distance between calibration points
    dist_px = np.nanmean(
        np.sqrt(
            np.square(pt_a_df["x"] - pt_b_df["x"])
            + np.square(pt_a_df["y"] - pt_b_df["y"]),
        ),
    )
    # Finding pixels per mm conversion using given width and height as calibration
    px_per_mm = dist_px / dist_mm
    # Saving to config file
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config.auto.px_per_mm = px_per_mm
    config_fp.write_text(config.model_dump_json(indent=2))


def _calc_exists_from_likelihood(
    keypoints_fp: Path,
    config_fp: Path,
) -> tuple[int, int]:
    """Determines whether subject exists.

    This is done by looking at a sliding window of time.
    If the median likelihood of the subject
    existing in each frame across the sliding window is
    greater than the defined pcutoff, then
    the determine this as the start time.

    Notes:
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
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config_filt = config.user.calculate_params.from_likelihood
    bpts = config.get_ref(config_filt.bodyparts)
    window_sec = config.get_ref(config_filt.window_sec)
    pcutoff = config.get_ref(config_filt.pcutoff)
    fps = config.auto.formatted_vid.fps
    assert fps != -1, (
        "fps not yet set. Please calculate fps first with `proj.get_vid_metadata`."
    )
    # Deriving more parameters
    window_frames = int(np.round(fps * window_sec, 0))
    # Loading dataframe
    keypoints_df = KeypointsDf.clean_headings(KeypointsDf.read(keypoints_fp))
    # Getting likehoods of subject (given bpts) existing in each frame
    KeypointsDf.check_bpts_exist(keypoints_df, bpts)
    idx = pd.IndexSlice
    lhood_df = pd.DataFrame(index=keypoints_df.index)
    indivs, _ = KeypointsDf.get_indivs_bpts(keypoints_df)
    for indiv in indivs:
        # Calculating likelihood of subject existing at each frame from median
        lhood_df[(indiv, "current")] = keypoints_df.loc[
            :,
            idx[indiv, bpts, LIKELIHOOD],
        ].apply(np.nanmedian, axis=1)
        # Calculating likelihood of subject existing over time window
        lhood_df[(indiv, "rolling")] = (
            lhood_df[(indiv, "current")]
            .rolling(window_frames, center=True)
            .agg(np.nanmean)
        )
    lhood_df.columns = pd.MultiIndex.from_tuples(lhood_df.columns)
    # Getting bool of frames where ALL indivs exist
    idx = pd.IndexSlice
    exists_vect = (lhood_df.loc[:, idx[:, "rolling"]] > pcutoff).all(axis=1)
    assert np.any(exists_vect), (
        "The subject was not detected in any frames. Please also check the video."
    )
    # Getting when subject first and last exists in video
    start_frame = lhood_df[exists_vect].index[0]
    stop_frame = lhood_df[exists_vect].index[-1]
    return start_frame, stop_frame
