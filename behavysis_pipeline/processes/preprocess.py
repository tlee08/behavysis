"""
Functions have the following format:

Parameters
----------
dlc_fp : str
    The input video filepath.
out_fp : str
    The output video filepath.
configs_fp : str
    The JSON configs filepath.
overwrite : bool
    Whether to overwrite the output file (if it exists).

Returns
-------
str
    Description of the function's outcome.

"""

import os

import numpy as np
import pandas as pd

from behavysis_pipeline.df_classes.keypoints_df import (
    Coords,
    IndivColumns,
    KeypointsDf,
)
from behavysis_pipeline.pydantic_models.configs import ExperimentConfigs
from behavysis_pipeline.utils.diagnostics_utils import file_exists_msg
from behavysis_pipeline.utils.logging_utils import get_io_obj_content, init_logger_with_io_obj
from behavysis_pipeline.utils.misc_utils import get_current_func_name


class Preprocess:
    """_summary_"""

    @classmethod
    def start_stop_trim(cls, dlc_fp: str, out_fp: str, configs_fp: str, overwrite: bool) -> str:
        """
        Filters the rows of a DLC formatted dataframe to include only rows within the start
        and end time of the experiment, given a corresponding configs dict.

        Parameters
        ----------
        dlc_fp : str
            The file path of the input DLC formatted dataframe.
        out_fp : str
            The file path of the output trimmed dataframe.
        configs_fp : str
            The file path of the configs dict.
        overwrite : bool
            If True, overwrite the output file if it already exists. If False, skip processing
            if the output file already exists.

        Returns
        -------
        str
            An outcome message indicating the result of the trimming process.

        Notes
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
        logger, io_obj = init_logger_with_io_obj(get_current_func_name())
        if not overwrite and os.path.exists(out_fp):
            logger.warning(file_exists_msg(out_fp))
            return get_io_obj_content(io_obj)
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        start_frame = configs.auto.start_frame
        stop_frame = configs.auto.stop_frame
        # Reading file
        df = KeypointsDf.read(dlc_fp)
        # Trimming dataframe between start and stop frames
        df = df.loc[start_frame:stop_frame, :]
        KeypointsDf.write(df, out_fp)
        return get_io_obj_content(io_obj)

    @classmethod
    def interpolate_stationary(cls, dlc_fp: str, out_fp: str, configs_fp: str, overwrite: bool) -> str:
        """
        If the point detection (above a certain threshold) is below a certain proportion, then the x and y coordinates are set to the given values (usually corners).
        Otherwise, does nothing (encouraged to run Preprocess.interpolate afterwards).

        Notes
        -----
        The config file must contain the following parameters:
        ```
        - user
            - preprocess
                - interpolate_stationary
                    [
                        - bodypart: str (assumed to be the "single" individual)
                        - pcutoff: float (between 0 and 1)
                        - pcutoff_all: float (between 0 and 1)
                        - x: float (between 0 and 1 - proportion of the video width)
                        - y: float (between 0 and 1 - proportion of the video height)
                    ]
        ```
        """
        logger, io_obj = init_logger_with_io_obj(get_current_func_name())
        if not overwrite and os.path.exists(out_fp):
            logger.warning(file_exists_msg(out_fp))
            return get_io_obj_content(io_obj)
        # Getting necessary config parameters list
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt_ls = configs.user.preprocess.interpolate_stationary
        # scorer = configs.auto.scorer_name
        width_px = configs.auto.formatted_vid.width_px
        height_px = configs.auto.formatted_vid.height_px
        if width_px is None or height_px is None:
            raise ValueError(
                "Width and height must be provided in the formatted video. Try running FormatVid.format_vid."
            )
        # Reading file
        df = KeypointsDf.read(dlc_fp)
        # Getting the scorer name
        scorer = df.columns.unique(KeypointsDf.CN.SCORER.value)[0]
        # For each bodypart, filling in the given point
        for configs_filt in configs_filt_ls:
            # Getting config parameters
            bodypart = configs_filt.bodypart
            pcutoff = configs_filt.pcutoff
            pcutoff_all = configs_filt.pcutoff_all
            x = configs_filt.x
            y = configs_filt.y
            # Converting x and y from video proportions to pixel coordinates
            x = x * width_px
            y = y * height_px
            # Getting "is_detected" for each frame for the bodypart
            is_detected = df[(scorer, "single", bodypart, Coords.LIKELIHOOD.value)] >= pcutoff
            # If the bodypart is detected in less than the given proportion of the video, then set the x and y coordinates to the given values
            if is_detected.mean() < pcutoff_all:
                df[(scorer, "single", bodypart, Coords.X.value)] = x
                df[(scorer, "single", bodypart, Coords.Y.value)] = y
                df[(scorer, "single", bodypart, Coords.LIKELIHOOD.value)] = pcutoff
                logger.info(
                    f"{bodypart} is detected in less than {pcutoff_all} of the video."
                    " Setting x and y coordinates to ({x}, {y})."
                )
            else:
                logger.info(
                    f"{bodypart} is detected in more than {pcutoff_all} of the video."
                    " No need for stationary interpolation."
                )
        # Saving
        KeypointsDf.write(df, out_fp)
        return get_io_obj_content(io_obj)

    @classmethod
    def interpolate(cls, dlc_fp: str, out_fp: str, configs_fp: str, overwrite: bool) -> str:
        """
        "Smooths" out noticeable jitter of points, where the likelihood (and accuracy) of
        a point's coordinates are low (e.g., when the subject's head goes out of view). It
        does this by linearly interpolating the frames of a body part that are below a given
        likelihood pcutoff.

        Notes
        -----
        The config file must contain the following parameters:
        ```
        - user
            - preprocess
                - interpolate
                    - pcutoff: float
        ```
        """
        logger, io_obj = init_logger_with_io_obj(get_current_func_name())
        if not overwrite and os.path.exists(out_fp):
            logger.warning(file_exists_msg(out_fp))
            return get_io_obj_content(io_obj)
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = configs.user.preprocess.interpolate
        # Reading file
        df = KeypointsDf.read(dlc_fp)
        # Gettings the unique groups of (individual, bodypart) groups.
        unique_cols = df.columns.droplevel(["coords"]).unique()
        # Setting low-likelihood points to Nan to later interpolate
        for scorer, indiv, bp in unique_cols:
            # Imputing Nan likelihood points with 0
            df[(scorer, indiv, bp, Coords.LIKELIHOOD.value)].fillna(value=0, inplace=True)
            # Setting x and y coordinates of points that have low likelihood to Nan
            to_remove = df[(scorer, indiv, bp, Coords.LIKELIHOOD.value)] < configs_filt.pcutoff
            df.loc[to_remove, (scorer, indiv, bp, Coords.X.value)] = np.nan
            df.loc[to_remove, (scorer, indiv, bp, Coords.Y.value)] = np.nan
        # linearly interpolating Nan x and y points.
        # Also backfilling points at the start.
        # Also forward filling points at the end.
        # Also imputing nan points with 0 (if the ENTIRE column is nan, then it's imputed)
        df = df.interpolate(method="linear").bfill().ffill()
        # if df.isnull().values.any() then the entire column is nan (print warning)
        df = df.fillna(0)
        KeypointsDf.write(df, out_fp)
        return get_io_obj_content(io_obj)

    @classmethod
    def refine_ids(cls, dlc_fp: str, out_fp: str, configs_fp: str, overwrite: bool) -> str:
        """
        Ensures that the identity is correctly tracked for maDLC.
        Assumes interpolate_points has already been run.

        Notes
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
        logger, io_obj = init_logger_with_io_obj(get_current_func_name())
        if not overwrite and os.path.exists(out_fp):
            logger.warning(file_exists_msg(out_fp))
            return get_io_obj_content(io_obj)
        # Reading file
        df = KeypointsDf.read(dlc_fp)
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = configs.user.preprocess.refine_ids
        marked = configs.get_ref(configs_filt.marked)
        unmarked = configs.get_ref(configs_filt.unmarked)
        marking = configs.get_ref(configs_filt.marking)
        window_sec = configs.get_ref(configs_filt.window_sec)
        bpts = configs.get_ref(configs_filt.bodyparts)
        metric = configs.get_ref(configs_filt.metric)
        fps = configs.auto.formatted_vid.fps
        # Calculating more parameters
        window_frames = int(np.round(fps * window_sec, 0))
        # Error checking for invalid/non-existent column names marked, unmarked, and marking
        for column, level in [
            (marked, "individuals"),
            (unmarked, "individuals"),
            (marking, "bodyparts"),
        ]:
            if column not in df.columns.unique(level):
                raise ValueError(
                    f'The marking value in the config file, "{column}",' " is not a column name in the DLC file."
                )
        # Checking that bodyparts are all valid
        KeypointsDf.check_bpts_exist(df, bpts)
        # Calculating the distances between the averaged bodycentres and the marking
        df_aggr = aggregate_df(df, marking, [marked, unmarked], bpts)
        # Getting "to_switch" decision series for each frame
        df_switch = decice_switch(df_aggr, window_frames, marked, unmarked)
        # Updating df with the switched values
        df_switched = switch_identities(df, df_switch[metric], marked, unmarked)
        KeypointsDf.write(df_switched, out_fp)
        return get_io_obj_content(io_obj)


def aggregate_df(
    df: pd.DataFrame,
    marking: str,
    indivs: list[str],
    bpts: list[str],
) -> pd.DataFrame:
    """
    _summary_

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    marking : str
        _description_
    indivs : list[str]
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    l0 = df.columns.unique(0)[0]
    df_aggr = pd.DataFrame(index=df.index)
    for coord in [Coords.X.value, Coords.Y.value]:
        # Getting the coordinates of the colour marking in each frame
        df_aggr[("mark", coord)] = df[l0, IndivColumns.SINGLE.value, marking, coord]
        idx = pd.IndexSlice
        for indiv in indivs:
            # Getting the coordinates of each individual (average of the given bodyparts list)
            df_aggr[(indiv, coord)] = df.loc[:, idx[l0, indiv, bpts, coord]].mean(axis=1)
    # Getting the Euclidean distance between each mouse and the colour marking in each frame
    for indiv in indivs:
        df_aggr[(indiv, "dist")] = np.sqrt(
            np.square(df_aggr[(indiv, Coords.X.value)] - df_aggr[("mark", Coords.X.value)])
            + np.square(df_aggr[(indiv, Coords.Y.value)] - df_aggr[("mark", Coords.Y.value)])
        )
    # Formatting columns as a MultiIndex
    df_aggr.columns = pd.MultiIndex.from_tuples(df_aggr.columns)
    return df_aggr


def decice_switch(
    df_aggr: pd.DataFrame,
    window_frames: int,
    marked: str,
    unmarked: str,
) -> pd.DataFrame:
    """
    Calculating different metrics for whether to swap the mice identities, depending
    on the current distance, rolling decision, and average binned decision.

    Parameters
    ----------
    df_aggr : pd.DataFrame
        _description_
    window_frames : int
        _description_
    marked : str
        _description_
    unmarked : str
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    df_switch = pd.DataFrame(index=df_aggr.index)
    #   - Current decision
    df_switch["current"] = df_aggr[(marked, "dist")] > df_aggr[(unmarked, "dist")]
    #   - Decision rolling
    df_switch["rolling"] = (
        df_switch["current"].rolling(window_frames, min_periods=1).apply(lambda x: x.mode()[0]).map({1: True, 0: False})
    )
    #   - Decision binned
    bins = np.arange(df_switch.index.min(), df_switch.index.max() + window_frames, window_frames)
    df_switch_x = pd.DataFrame()
    df_switch_x["bins"] = pd.Series(pd.cut(df_switch.index, bins=bins, labels=bins[1:], include_lowest=True))
    df_switch_x["current"] = df_switch["current"]
    df_switch["binned"] = df_switch_x.groupby("bins")["current"].transform(lambda x: x.mode())
    return df_switch


def switch_identities(
    df: pd.DataFrame,
    is_switch: pd.Series,
    marked: str,
    unmarked: str,
) -> pd.DataFrame:
    """
    _summary_

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    isSwitch : pd.Series
        _description_
    marked : str
        _description_
    unmarked : str
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    df = df.copy()
    header = df.columns.unique(0)[0]
    df["isSwitch"] = is_switch

    def _f(row: pd.Series, marked: str, unmarked: str) -> pd.Series:
        if row["isSwitch"][0]:
            temp = list(row.loc[header, unmarked].copy())
            row[header, unmarked] = list(row[header, marked].copy())
            row[header, marked] = temp
        return row

    df = df.apply(lambda row: _f(row, marked, unmarked), axis=1)
    df = df.drop(columns="isSwitch")
    return df
