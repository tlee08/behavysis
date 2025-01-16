"""
Utility functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mode

from behavysis_pipeline.df_classes.behav_df import BehavColumns, BehavDf
from behavysis_pipeline.mixins.misc_mixin import MiscMixin
from behavysis_pipeline.pydantic_models.bouts import Bouts

####################################################################################################
# DF CLASS
####################################################################################################


class BoutsDf(BehavDf):
    """
    Mixin for behaviour DF
    (generated from behavysis behaviour classification)
    functions.
    NOTE: not actually a DF mixin but used to convert to and from fbf df and bouts json format
    TODO: is this structure correct?
    """

    @staticmethod
    def vect2bouts(vect: np.ndarray | pd.Series) -> pd.DataFrame:
        """
        Will return a dataframe with the start and stop indexes of each contiguous set of
        positive values (i.e. a bout).

        Parameters
        ----------
        vect : np.ndarray | pd.Series
            Expects a vector of booleans

        Returns
        -------
        pd.DataFrame
            _description_
        """
        offset = 0
        if isinstance(vect, pd.Series):
            if vect.shape[0] > 0:
                offset = vect.index[0]
        # Getting stop and start indexes of each bout
        z = np.concatenate(([0], vect, [0]))
        start = np.flatnonzero(~z[:-1] & z[1:])
        stop = np.flatnonzero(z[:-1] & ~z[1:]) - 1
        bouts_ls = np.column_stack((start, stop)) + offset
        # Making dataframe
        bouts_df = pd.DataFrame(bouts_ls, columns=["start", "stop"])
        bouts_df["dur"] = bouts_df["stop"] - bouts_df["start"] + 1
        return bouts_df

    @classmethod
    def frames2bouts(cls, frames_df: pd.DataFrame) -> Bouts:
        """
        Frames df to bouts model object.
        """
        bouts_ls = []
        # For each behaviour
        for behav in frames_df.columns.unique(BehavDf.CN.BEHAVIOURS.value):
            # Getting start-stop of each bout
            start_stop_df = cls.vect2bouts(frames_df[(behav, "pred")])
            # For each bout (i.e. start-stop pair)
            for _, row in start_stop_df.iterrows():
                # Getting only the frames in the current bout
                bout_frames_df = frames_df.loc[row["start"] : row["stop"]]
                # Preparing to make Bout model object
                bout_dict = {
                    "start": row["start"],
                    "stop": row["stop"],
                    "behaviour": behav,
                    "actual": int(mode(bout_frames_df[(behav, BehavColumns.ACTUAL.value)]).mode),
                    "user_defined": {},
                }
                # Getting the mode value for the bout (actual, and specific user_behavs)
                for outcome, values in bout_frames_df[behav].items():
                    if outcome not in MiscMixin.enum2tuple(BehavColumns):
                        bout_dict["user_defined"][str(outcome)] = int(mode(values).mode)
                # Making the Bout model object and appending to bouts_ls
                bouts_ls.append(bout_dict)
        # Making and return the Bouts model object
        return Bouts(start=frames_df.index[0], stop=frames_df.index[-1] + 1, bouts=bouts_ls)

    @classmethod
    def include_outcome_behavs(cls, df: pd.DataFrame, behav_outcomes: dict[str, tuple[str]]) -> pd.DataFrame:
        """
        Adding the user_behavs columns to the df for each behaviour.

        Expects `behav_outcomes` to be a dictionary with the behaviour str
        as the key and a tuple of specific behaviour strs as the value.
        """
        # Keeping the `actual`, `pred`, and all user_behavs columns
        out_df = cls.init_df(df.index)
        a = BehavColumns.ACTUAL.value
        p = BehavColumns.PRED.value
        for behav, user_behavs in behav_outcomes.items():
            # Adding pred column
            out_df[(behav, p)] = df[(behav, p)].values
            # Adding actual column
            # TODO: quick flip but make more explicit
            out_df[(behav, a)] = df[(behav, p)].values * np.array(-1)
            # Adding user_behav columns
            for i in user_behavs:
                out_df[(behav, i)] = 0
        # Ordering by "behaviours" level
        out_df = out_df.sort_index(axis=1, level=BehavDf.CN.BEHAVIOURS.value)
        # Returning the new df
        return out_df

    @classmethod
    def bouts2frames(cls, bouts: Bouts) -> pd.DataFrame:
        """
        Bouts model object to frames df.
        """
        # Making columns
        all_behavs = {}  # behav: user_behav_ls pairs
        for bout in bouts.bouts:
            if bout.behaviour not in all_behavs:
                all_behavs[bout.behaviour] = {
                    BehavColumns.PRED.value,
                    BehavColumns.ACTUAL.value,
                }
            all_behavs[bout.behaviour] |= set(bout.user_defined.keys())

        # construct ret_df with index from given start and stop, and all_behavs dict
        ret_df = cls.init_df(pd.Series(np.arange(bouts.start, bouts.stop)))
        for behav, outcomes in all_behavs.items():
            for outcome in outcomes:
                ret_df[(behav, outcome)] = 0
        ret_df = ret_df.sort_index(axis=1)
        # Filling in all user_behavs columns for each behaviour
        for bout in bouts.bouts:
            bout_ret_df = ret_df.loc[bout.start : bout.stop]
            # Filling in predicted behaviour column
            bout_ret_df.loc[:, (bout.behaviour, BehavColumns.PRED.value)] = 1
            # Filling in actual behaviour column
            bout_ret_df.loc[:, (bout.behaviour, BehavColumns.ACTUAL.value)] = bout.actual
            # Filling in user_behavs columns
            for k, v in bout.user_defined.items():
                bout_ret_df.loc[:, (bout.behaviour, k)] = v
        # Returning frames df
        return ret_df
