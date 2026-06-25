"""Functions have the following format.

Parameters
----------
dlc_fp : str
    The input video filepath.
dst_fp : str
    The output video filepath.
config_fp : str
    The JSON config filepath.
overwrite : bool
    Whether to overwrite the output file (if it exists).

Returns:
-------
str
    Description of the function's outcome.

"""

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from loguru import logger

from behavysis.constants import LIKELIHOOD, SCORER, SINGLE, X, Y
from behavysis.df_classes import KeypointsDf
from behavysis.models import ExperimentConfig
from behavysis.utils.io_utils import file_exists_msg


class PreprocessFunc(Protocol):
    """Protocol for preprocess functions."""

    def __call__(
        self,
        src_fp: Path,
        dst_fp: Path,
        config_fp: Path,
        *,
        overwrite: bool,
    ) -> None:
        """Protocol for preprocess functions."""


def start_stop_trim(
    src_fp: Path,
    dst_fp: Path,
    config_fp: Path,
    *,
    overwrite: bool,
) -> None:
    """Filters the rows of a DLC formatted dataframe.

    Includes only rows within the start
    and end time of the experiment, given a corresponding config dict.

    Parameters
    ----------
    dlc_fp : str
        The file path of the input DLC formatted dataframe.
    dst_fp : Path
        The file path of the output trimmed dataframe.
    config_fp : Path
        The file path of the config dict.
    overwrite : bool
        If True, overwrite the output file if it already exists. If False, skip
        if the output file already exists.

    Returns:
    -------
    str
        An outcome message indicating the result of the trimming process.

    Notes:
    -----
    The config file must contain the following parameters:
    ```
    - user
        - preprocess
            - start_stop_trim
                - start_frame: int
                - stop_frame: int
    ```
    """
    if not overwrite and dst_fp.exists():
        logger.warning(file_exists_msg(dst_fp))
        return
    # Getting necessary config parameters
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    start_frame = config.auto.start_frame
    stop_frame = config.auto.stop_frame
    # Reading file
    keypoints_df = KeypointsDf.read(src_fp)
    # Trimming dataframe between start and stop frames
    keypoints_df = keypoints_df.loc[start_frame:stop_frame, :]
    KeypointsDf.write(keypoints_df, dst_fp)


def interpolate_stationary(
    src_fp: Path,
    dst_fp: Path,
    config_fp: Path,
    *,
    overwrite: bool,
) -> None:
    """If the point detection (above a certain threshold) is below a certain proportion.

    Then the x and y coordinates are set to the given values (usually corners).
    Otherwise, does nothing (encouraged to run Preprocess.interpolate afterwards).

    """
    if not overwrite and dst_fp.exists():
        logger.warning(file_exists_msg(dst_fp))
        return
    # Getting necessary config parameters list
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config_filt_ls = config.user.preprocess.interpolate_stationary
    width_px = config.auto.formatted_vid.width_px
    height_px = config.auto.formatted_vid.height_px
    if width_px <= 0 or height_px <= 0:
        msg = (
            f"Video dimensions not set for experiment.\n"
            f"  width_px={width_px}, height_px={height_px}\n"
            f"  Run proj.format_video() first to set these values."
        )
        raise ValueError(msg)
    # Reading file
    keypoints_df = KeypointsDf.read(src_fp)
    # Getting the scorer name
    scorer = keypoints_df.columns.unique(SCORER)[0]
    # For each bodypart, filling in the given point
    for config_filt in config_filt_ls:
        # Getting config parameters
        bodypart = config_filt.bodypart
        pcutoff = config_filt.pcutoff
        pcutoff_all = config_filt.pcutoff_all
        x = config_filt.x
        y = config_filt.y
        # Converting x and y from video proportions to pixel coordinates
        x = x * width_px
        y = y * height_px
        # Getting "is_detected" for each frame for the bodypart
        is_detected = keypoints_df[(scorer, "single", bodypart, LIKELIHOOD)] >= pcutoff
        # If the bodypart is detected in less than the given proportion of the video,
        # then set the x and y coordinates to the given values
        if is_detected.mean() < pcutoff_all:
            keypoints_df[(scorer, "single", bodypart, X)] = x
            keypoints_df[(scorer, "single", bodypart, Y)] = y
            keypoints_df[(scorer, "single", bodypart, LIKELIHOOD)] = pcutoff
            logger.info(
                f"{bodypart} is detected in less than {pcutoff_all} of the video."
                f" Setting x and y coordinates to ({x}, {y})."
            )
        else:
            logger.info(
                f"{bodypart} is detected in more than {pcutoff_all} of the video."
                " No need for stationary interpolation."
            )
    # Saving
    KeypointsDf.write(keypoints_df, dst_fp)


def interpolate(
    src_fp: Path,
    dst_fp: Path,
    config_fp: Path,
    *,
    overwrite: bool,
) -> None:
    """Smooths noticeable jitter of points.

    Where the likelihood (and accuracy) of
    a point's coordinates are low
    (e.g., when the subject's head goes out of view).
    It does this by linearly interpolating the frames
    of a body part that are below a given likelihood pcutoff.

    Notes:
    -----
    The config file must contain the following parameters:
    ```
    - user
        - preprocess
            - interpolate
                - pcutoff: float
    ```
    """
    # Have error checking for any columns that have NO points above the pcutoff
    # (so they are all NaN)
    if not overwrite and dst_fp.exists():
        logger.warning(file_exists_msg(dst_fp))
        return
    # Getting necessary config parameters
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config_filt = config.user.preprocess.interpolate
    # Reading file
    keypoints_df = KeypointsDf.read(src_fp)
    # Gettings the unique groups of (individual, bodypart) groups.
    unique_cols = keypoints_df.columns.droplevel(["coords"]).unique()
    # Setting low-likelihood points to Nan to later interpolate
    for scorer, indiv, bp in unique_cols:
        # Imputing Nan likelihood points with 0
        keypoints_df[(scorer, indiv, bp, LIKELIHOOD)] = keypoints_df[
            (scorer, indiv, bp, LIKELIHOOD)
        ].fillna(value=0)
        # Setting x and y coordinates of points that have low likelihood to Nan
        to_remove = keypoints_df[(scorer, indiv, bp, LIKELIHOOD)] < config_filt.pcutoff
        keypoints_df.loc[to_remove, (scorer, indiv, bp, X)] = np.nan
        keypoints_df.loc[to_remove, (scorer, indiv, bp, Y)] = np.nan
    # linearly interpolating Nan x and y points.
    # Also backfilling points at the start.
    # Also forward filling points at the end.
    # Also imputing nan points with 0 (if the ENTIRE column is nan, then it's imputed)
    keypoints_df = keypoints_df.interpolate(method="linear").bfill().ffill()
    # if df.isna().to_numpy().any() then the entire column is nan (log warning)
    KeypointsDf.write(keypoints_df, dst_fp)


def refine_ids(src_fp: Path, dst_fp: Path, config_fp: Path, *, overwrite: bool) -> None:
    """Ensures that the identity is correctly tracked for maDLC.

    Assumes interpolate_points has already been run.

    Notes:
    -----
    The config file must contain the following parameters:
    ```
    - user
        - preprocess
            - refine_ids
                - marked: str
                - unmarked: str
                - marking: str
                - window_sec: float
                - metric: ["current", "rolling", "binned"]
    ```
    """
    if not overwrite and dst_fp.exists():
        logger.warning(file_exists_msg(dst_fp))
        return
    # Reading file
    keypoints_df = KeypointsDf.read(src_fp)
    # Getting necessary config parameters
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config_filt = config.user.preprocess.refine_ids
    marked = config.get_ref(config_filt.marked)
    unmarked = config.get_ref(config_filt.unmarked)
    marking = config.get_ref(config_filt.marking)
    window_sec = config.get_ref(config_filt.window_sec)
    bpts = config.get_ref(config_filt.bodyparts)
    metric = config.get_ref(config_filt.metric)
    fps = config.auto.formatted_vid.fps
    # Calculating more parameters
    window_frames = int(np.round(fps * window_sec, 0))
    # Error checking for invalid/non-existent column names marked, unmarked, and marking
    for column, level in [
        (marked, "individuals"),
        (unmarked, "individuals"),
        (marking, "bodyparts"),
    ]:
        if column not in keypoints_df.columns.unique(level):
            msg = (
                f'The marking value in the config file, "{column}'
                "is not a column name in the DLC file."
            )
            raise ValueError(msg)
    # Checking that bodyparts are all valid
    KeypointsDf.check_bpts_exist(keypoints_df, bpts)
    # Calculating the distances between the averaged bodycentres and the marking
    mark_dists_df = _get_mark_dists_df(keypoints_df, marked, unmarked, [marking], bpts)
    # Getting "to_switch" decision series for each frame
    switch_df = _get_id_switch_df(mark_dists_df, window_frames, marked, unmarked)
    # Updating df with the switched values
    switched_keypoints_df = _switch_identities(
        keypoints_df, switch_df[metric], marked, unmarked
    )
    KeypointsDf.write(switched_keypoints_df, dst_fp)


def _get_mark_dists_df(
    keypoints_df: pd.DataFrame,
    marked_indiv: str,
    unmarked_indiv: str,
    mark_pts: list[str],
    bpts: list[str],
) -> pd.DataFrame:
    """Calculate distances between individuals and marking for identity refinement.

    Parameters
    ----------
    keypoints_df : pd.DataFrame
        Keypoints dataframe with coordinates.
    marked_indiv : str
        Name of marked individual.
    unmarked_indiv : str
        Name of unmarked individual.
    mark_pts : list[str]
        Marking bodypoints.
    bpts : list[str]
        Bodypoints to average for distance calculation.

    Returns:
    -------
    pd.DataFrame
        DataFrame with distances to marking for each individual.
    """
    l0 = keypoints_df.columns.unique(0)[0]
    mark_dists_df = pd.DataFrame(index=keypoints_df.index)
    indivs = [marked_indiv, unmarked_indiv]
    for coord in [X, Y]:
        idx = pd.IndexSlice
        # Getting the coordinates of the colour marking in each frame
        mark_dists_df[("mark", coord)] = keypoints_df.loc[
            :, idx[l0, SINGLE, mark_pts, coord]
        ].mean(axis=1)
        for indiv in indivs:
            # Getting the coordinates of each individual (average of the bpts list)
            mark_dists_df[(indiv, coord)] = keypoints_df.loc[
                :, idx[l0, indiv, bpts, coord]
            ].mean(axis=1)
    # Getting the Euclidean distance between each mouse
    # and the colour marking in each frame
    for indiv in indivs:
        mark_dists_df[(indiv, "dist")] = np.sqrt(
            np.square(mark_dists_df[(indiv, X)] - mark_dists_df[("mark", X)])
            + np.square(mark_dists_df[(indiv, Y)] - mark_dists_df[("mark", Y)])
        )
    # Formatting columns as a MultiIndex
    mark_dists_df.columns = pd.MultiIndex.from_tuples(mark_dists_df.columns)
    return mark_dists_df


def _get_id_switch_df(
    mark_dists_df: pd.DataFrame,
    window_frames: int,
    marked: str,
    unmarked: str,
) -> pd.DataFrame:
    """Calculate identity switch decisions using current, rolling, and binned metrics.

    Parameters
    ----------
    mark_dists_df : pd.DataFrame
        DataFrame with distances to marking for each individual.
    window_frames : int
        Window size in frames for rolling/binned calculations.
    marked : str
        Name of marked individual.
    unmarked : str
        Name of unmarked individual.

    Returns:
    -------
    pd.DataFrame
        DataFrame with 'current', 'rolling', and 'binned' switch decisions.
    """
    switch_df = pd.DataFrame(index=mark_dists_df.index)
    #   - Current decision
    switch_df["current"] = (
        mark_dists_df[(marked, "dist")] > mark_dists_df[(unmarked, "dist")]
    )
    #   - Decision rolling
    switch_df["rolling"] = (
        switch_df["current"]
        .rolling(window_frames, min_periods=1)
        .apply(lambda x: x.mode()[0])
        .map({1: True, 0: False})
    )
    #   - Decision binned
    bins = np.arange(
        switch_df.index.min(), switch_df.index.max() + window_frames, window_frames
    )
    df_switch_x = pd.DataFrame()
    df_switch_x["bins"] = pd.Series(
        pd.cut(switch_df.index, bins=bins, labels=bins[1:], include_lowest=True)
    )
    df_switch_x["current"] = switch_df["current"]
    switch_df["binned"] = df_switch_x.groupby("bins")["current"].transform(
        lambda x: x.mode()
    )
    return switch_df


def _switch_identities(
    keypoints_df: pd.DataFrame,
    is_switch: pd.Series,
    marked_indiv: str,
    unmarked_indiv: str,
) -> pd.DataFrame:
    """Swap individual identities in keypoints dataframe where is_switch is True.

    Parameters
    ----------
    keypoints_df : pd.DataFrame
        Keypoints dataframe with individual columns.
    is_switch : pd.Series
        Boolean series indicating which frames need identity swap.
    marked_indiv : str
        Name of marked individual.
    unmarked_indiv : str
        Name of unmarked individual.

    Returns:
    -------
    pd.DataFrame
        Keypoints dataframe with swapped identities.
    """
    keypoints_df = keypoints_df.copy()
    header = keypoints_df.columns.unique(0)[0]
    keypoints_df["isSwitch"] = is_switch

    def _f(row: pd.Series, marked: str, unmarked: str) -> pd.Series:
        if row["isSwitch"][0]:
            temp = list(row.loc[header, unmarked].copy())
            row[header, unmarked] = list(row[header, marked].copy())
            row[header, marked] = temp
        return row

    keypoints_df = keypoints_df.apply(
        lambda row: _f(row, marked_indiv, unmarked_indiv), axis=1
    )
    keypoints_df = keypoints_df.drop(columns="isSwitch")
    return keypoints_df
