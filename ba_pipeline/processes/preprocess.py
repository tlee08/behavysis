"""
Functions have the following format:

Parameters
----------
in_fp : str
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
from ba_core.data_models.experiment_configs import ExperimentConfigs
from ba_core.mixins.df_io_mixin import DFIOMixin
from ba_core.mixins.diagnostics_mixin import DiagnosticsMixin
from ba_core.mixins.keypoints_mixin import KeypointsMixin
from ba_core.utils.constants import BODYCENTRE, SINGLE_COL


class Preprocess:
    """_summary_"""

    @staticmethod
    def start_stop_trim(
        in_fp: str,
        out_fp: str,
        configs_fp: str,
        overwrite: bool,
    ) -> str:
        """
        Filters the rows of a DLC formatted dataframe to include only rows within the start
        and end time of the experiment, given a corresponding configs dict.

        Notes
        -----
        The config file must contain the following parameters:
        ```
        - (user, auto)
            - preprocess
                - start_stop_trim
                    - start_frame: int
                    - stop_frame: int
        ```
        """
        outcome = ""
        # If overwrite is False, checking if we should skip processing
        if not overwrite and os.path.exists(out_fp):
            return DiagnosticsMixin.warning_msg()
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        start_frame = configs.auto.start_frame
        stop_frame = configs.auto.stop_frame
        # Reading file
        df = DFIOMixin.read_feather(in_fp)
        # Trimming dataframe
        df = df.loc[start_frame:stop_frame, :]
        # Writing file
        DFIOMixin.write_feather(df, out_fp)
        return outcome

    @staticmethod
    def interpolate_points(
        in_fp: str,
        out_fp: str,
        configs_fp: str,
        overwrite: bool,
    ) -> str:
        """
        "Smooths" out noticeable jitter of points, where the likelihood (and accuracy) of
        a point's coordinates are low (e.g., when the subject's head goes out of view). It
        does this by linearly interpolating the frames of a body part that are below a given
        likelihood pcutoff.

        Notes
        -----
        The config file must contain the following parameters:
        ```
        - (user, auto)
            - preprocess
                - interpolate_points
                    - pcutoff: float
        ```
        """
        outcome = ""
        # If overwrite is False, checking if we should skip processing
        if not overwrite and os.path.exists(out_fp):
            return DiagnosticsMixin.warning_msg()
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = configs.user.preprocess.interpolate_points
        pcutoff = configs_filt.pcutoff
        # Reading file
        df = DFIOMixin.read_feather(in_fp)
        # Gettings the unique groups of (individual, bodypart) groups.
        unique_cols = df.columns.droplevel(["coords"]).unique()
        # Setting low-likelihood points to Nan to later interpolate
        for scorer, indiv, bp in unique_cols:
            try:
                # Imputing Nan likelihood points with 0
                df[(scorer, indiv, bp, "likelihood")].fillna(value=0, inplace=True)
                # Setting x and y coordinates of points that have low likelihood to Nan
                to_remove = df[(scorer, indiv, bp, "likelihood")] < pcutoff
                df.loc[to_remove, (scorer, indiv, bp, "x")] = np.nan
                df.loc[to_remove, (scorer, indiv, bp, "y")] = np.nan
            except KeyError:
                pass
        # linearly interpolating Nan x and y points. Also backfilling points at the start.
        df = df.interpolate(method="linear", axis=0).bfill()
        # Writing file
        DFIOMixin.write_feather(df, out_fp)
        return outcome

    @staticmethod
    def bodycentre(
        in_fp: str,
        out_fp: str,
        configs_fp: str,
        overwrite: bool,
    ) -> str:
        """
        Calculates the body centre of the subject in each frame, given the body parts to
        consider in the configs. The calculation used is the mean of all the body parts to consider.
        A new column is added called "name", with "x" and "y" sub-columns (does not include a
        "likelihood" column).

        Notes
        -----
        The config file must contain the following parameters:
        ```
        - (user, auto)
            - preprocess
                - interpolate_points
                    - pcutoff: float
        ```
        """
        outcome = ""
        # If overwrite is False, checking if we should skip processing
        if not overwrite and os.path.exists(out_fp):
            return DiagnosticsMixin.warning_msg()
        # Reading file
        df = DFIOMixin.read_feather(in_fp)
        # Getting indivs and bpts list
        indivs, _ = KeypointsMixin.get_headings(df)
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = configs.user.preprocess.bodycentre
        bpts = configs_filt.bodyparts
        # Checking that the bodyparts are all valid
        KeypointsMixin.check_bpts_exist(df, bpts)
        # Calculating the body centre in each frame for each individual
        l0 = df.columns.unique(0)[0]
        idx = pd.IndexSlice
        for indiv in indivs:
            for coord in ["x", "y", "likelihood"]:
                df[l0, indiv, BODYCENTRE, coord] = df.loc[
                    :, idx[l0, indiv, bpts, coord]
                ].apply(np.nanmean, axis=1)
        # Writing file
        DFIOMixin.write_feather(df, out_fp)
        return outcome

    @staticmethod
    def refine_identities(
        in_fp: str, out_fp: str, configs_fp: str, overwrite: bool
    ) -> str:
        """
        Ensures that the identity is correctly tracked for maDLC.
        Assumes interpolatePoints and calcBodyCentre has already been run.

        Notes
        -----
        The config file must contain the following parameters:
        ```
        - (user, auto)
            - preprocess
                - refine_identities
                    - marked: str
                    - unmarked: str
                    - marking: str
                    - window_sec: float
                    - metric: ["current", "rolling", "binned"]
        ```
        """
        outcome = ""
        # If overwrite is False, checking if we should skip processing
        if not overwrite and os.path.exists(out_fp):
            return DiagnosticsMixin.warning_msg()
        # Reading file
        df = DFIOMixin.read_feather(in_fp)
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = configs.user.preprocess.refine_identities
        # bpts = (
        #     configs_filt.bodyparts
        # )  # TODO: need to implement the body_centre function again
        marked = configs_filt.marked
        unmarked = configs_filt.unmarked
        marking = configs_filt.marking
        window_sec = configs_filt.window_sec
        metric = configs_filt.metric
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
                    f'The marking value in the config file, "{column}",'
                    + " is not a column name in the DLC file."
                )
        # Calculating the distances between the bodycentres and the marking
        df_aggr = aggregate_df(df, marking, [marked, unmarked])
        # Getting "to_switch" decision series for each frame
        df_switch = decice_switch(df_aggr, window_frames, marked, unmarked)
        # Updating df with the switched values
        df_switched = switch_identities(df, df_switch[metric], marked, unmarked)
        # Writing to file
        DFIOMixin.write_feather(df_switched, out_fp)
        return outcome


def aggregate_df(
    df: pd.DataFrame,
    marking: str,
    indivs: list[str],
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
    for coord in ["x", "y"]:
        # Getting the coordinates of the colour marking in each frame
        df_aggr[("mark", coord)] = df[l0, SINGLE_COL, marking, coord]
        for indiv in indivs:
            # Getting the coordinates of each individual (average of the given bodyparts list)
            df_aggr[(indiv, coord)] = df[l0, indiv, BODYCENTRE, coord]
            df_aggr[(indiv, coord)] = df[l0, indiv, BODYCENTRE, coord]
    # Getting the distance between each mouse and the colour marking in each frame
    for indiv in indivs:
        df_aggr[(indiv, "dist")] = np.sqrt(
            np.square(df_aggr[(indiv, "x")] - df_aggr[("mark", "x")])
            + np.square(df_aggr[(indiv, "y")] - df_aggr[("mark", "y")])
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
        df_switch["current"]
        .rolling(window_frames, min_periods=1)
        .apply(lambda x: x.mode()[0])
        .map({1: True, 0: False})
    )
    #   - Decision binned
    bins = np.arange(
        df_switch.index.min(), df_switch.index.max() + window_frames, window_frames
    )
    df_switch_x = pd.DataFrame()
    df_switch_x["bins"] = pd.Series(
        pd.cut(df_switch.index, bins=bins, labels=bins[1:], include_lowest=True)
    )
    df_switch_x["current"] = df_switch["current"]
    df_switch["binned"] = df_switch_x.groupby("bins")["current"].transform(
        lambda x: x.mode()
    )
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
