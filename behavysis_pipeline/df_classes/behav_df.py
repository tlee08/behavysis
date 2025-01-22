"""
Utility functions.
"""

from enum import Enum
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import mode

from behavysis_pipeline.df_classes.df_mixin import DFMixin, FramesIN
from behavysis_pipeline.pydantic_models.bouts import Bout, Bouts, BoutStruct
from behavysis_pipeline.utils.misc_utils import enum2tuple

####################################################################################################
# DF CONSTANTS
####################################################################################################


class OutcomesPredictedCols(Enum):
    PROB = "prob"
    PRED = "pred"


class OutcomesScoredCols(Enum):
    PROB = "prob"
    PRED = "pred"
    ACTUAL = "actual"


class BehavValues(Enum):
    BEHAV = 1
    NON_BEHAV = 0
    UNDETERMINED = -1


class BoutCols(Enum):
    START = "start"
    STOP = "stop"
    DUR = "dur"
    BEHAV = "behav"
    ACTUAL = "actual"
    USER_DEFINED = "user_defined"


class BehavCN(Enum):
    BEHAVS = "behavs"
    OUTCOMES = "outcomes"


####################################################################################################
# DF CLASS
####################################################################################################


class BehavDf(DFMixin):
    """
    Mixin for behaviour DF
    (generated from maDLC keypoint detection)
    functions.
    """

    NULLABLE = False
    IN = FramesIN
    CN = BehavCN

    OutcomesCols = None

    @classmethod
    def update_behav(cls, df: pd.DataFrame, behav_src: str, behav_dst: str) -> pd.DataFrame:
        """
        Update the given behaviour name (behav_src) with a new name (behav_dst).
        """
        # Getting columns
        columns = df.columns.to_frame(index=False)
        # Updating the behaviour column
        columns[cls.CN.BEHAVS.value] = columns[cls.CN.OUTCOMES.value].map(lambda x: behav_dst if x == behav_src else x)
        # Setting the new columns
        df.columns = pd.MultiIndex.from_frame(columns)
        # Returning
        return df

    @classmethod
    def check_df(cls, df: pd.DataFrame) -> None:
        """
        Checking the dataframe.
        """
        super().check_df(df)
        cls.check_outcomes_cols(df)

    @classmethod
    def check_outcomes_cols(cls, df: pd.DataFrame) -> None:
        """Asserting that, for each behaviour, the outcomes columns are present."""
        if cls.OutcomesCols:
            columns = df.columns
            outcomes_cols_ls = enum2tuple(cls.OutcomesCols)
            # For each behaviour
            for behav in columns.unique(cls.CN.BEHAVS.value):
                # For each outcome
                for outcome in outcomes_cols_ls:
                    # Assert the (behav, outcome) column is present
                    assert (behav, outcome) in columns, (
                        f"Expected {outcome} column for {behav}.\n" f"Only columns are: {columns}"
                    )


class BehavPredictedDf(BehavDf):
    """
    Mixin for behaviour DF, specifically for predicted behaviours dfs.
    """

    OutcomesCols = OutcomesPredictedCols


class BehavScoredDf(BehavDf):
    """
    Mixin for behaviour DF, specifically for scored behaviours dfs.
    """

    OutcomesCols = OutcomesScoredCols
    BoutCols = BoutCols

    @classmethod
    def import_boris_tsv(cls, fp: str, behavs_ls: list[str], start_frame: int, stop_frame: int) -> pd.DataFrame:
        """
        Importing Boris TSV file.
        """
        # Making df structure
        df = cls.init_df(pd.Series(np.arange(start_frame, stop_frame)))
        # Reading in corresponding BORIS tsv file
        df_boris = pd.read_csv(fp, sep="\t")
        # Initialising new classification columns based on
        # BORIS behavs and given `behavs_ls`
        # TODO: how to reconcile this with the behavs_ls?
        for behav in df_boris["Behavior"].unique():
            df[(behav, cls.OutcomesCols.ACTUAL.value)] = 0
            df[(behav, cls.OutcomesCols.PRED.value)] = 0
        for behav in behavs_ls:
            df[(behav, cls.OutcomesCols.ACTUAL.value)] = 0
            df[(behav, cls.OutcomesCols.PRED.value)] = 0
        # Setting the classification values from the BORIS file
        for ind, row in df_boris.iterrows():
            # Getting corresponding frame of this event point
            behav = row["Behavior"]
            frame = row["Image index"]
            status = row["Behavior type"]
            # Updating the classification in the scored df
            df.loc[frame:, (behav, cls.OutcomesCols.ACTUAL.value)] = status == "START"
            df.loc[frame:, (behav, cls.OutcomesCols.PRED.value)] = status == "START"
        # Setting dtype to int8
        df = df.astype(np.int8)
        return df

    ###############################################################################################
    # CONVERT FROM PREDICTED TO SCORED BEHAV DF
    ###############################################################################################

    @classmethod
    def get_bouts_struct_from_df(cls, df: pd.DataFrame) -> List[BoutStruct]:
        """
        Returns the list BoutStruct objects from the given BehavDf's columns.
        """
        bouts_struct = []
        for behav in df.columns.unique(cls.CN.BEHAVS.value):
            bouts_struct.append(
                BoutStruct.model_validate(
                    {
                        cls.BoutCols.BEHAV.value: behav,
                        cls.BoutCols.USER_DEFINED.value: list(df[behav].columns.unique(cls.CN.OUTCOMES.value)),
                    }
                )
            )
        return bouts_struct

    @classmethod
    def predicted2scored(cls, df: pd.DataFrame, bouts_struct: List[BoutStruct] | None = None) -> pd.DataFrame:
        """
        Convert a predicted behaviours dataframe to a scored behaviours dataframe.
        """
        # If behav_outcomes_ls is None, then initialising it from df
        bouts_struct = bouts_struct or cls.get_bouts_struct_from_df(df)
        # Making a new df
        scored_df = cls.init_df(df.index)
        # For each behaviour in behav_outcomes_ls
        for bout_struct in bouts_struct:
            behav = bout_struct.behav
            user_defined = bout_struct.user_defined
            # Adding pred column
            scored_df[(behav, cls.OutcomesCols.PRED.value)] = df[(behav, cls.OutcomesCols.PRED.value)].values
            # Adding actual column
            # NOTE: all predicted behav is set as undetermined in "actual" column
            scored_df[(behav, cls.OutcomesCols.ACTUAL.value)] = scored_df[(behav, cls.OutcomesCols.PRED.value)].where(
                cond=scored_df[(behav, cls.OutcomesCols.PRED.value)].values == BehavValues.BEHAV.value,
                other=BehavValues.UNDETERMINED.value,
            )
            # Adding user_defined columns and setting values to 0
            for user_defined_i in user_defined:
                scored_df[(behav, user_defined_i)] = 0
        # Ordering by BEHAVS level
        scored_df = scored_df.sort_index(axis=1, level=cls.CN.BEHAVS.value)
        # Checking
        cls.check_df(scored_df)
        # Returning
        return scored_df

    ###############################################################################################
    # BOUTS FUNCS
    ###############################################################################################

    @classmethod
    def vect2bouts(
        cls,
        vect: np.ndarray | pd.Series,
    ) -> pd.DataFrame:
        """
        Will return a dataframe with the start and stop indexes of each contiguous set of
        positive/true values (i.e. a bout).

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
        # Making dataframe
        bouts_df = pd.DataFrame({cls.BoutCols.START.value: start, cls.BoutCols.STOP.value: stop}) + offset
        bouts_df[cls.BoutCols.DUR.value] = bouts_df[cls.BoutCols.STOP.value] - bouts_df[cls.BoutCols.START.value] + 1
        return bouts_df

    @classmethod
    def frames2bouts(cls, df: pd.DataFrame) -> Bouts:
        """
        Frames df to bouts model object.
        """
        # Getting bouts_ls
        bouts_ls = []
        # For each behaviour
        for behav in df.columns.unique(cls.CN.BEHAVS.value):
            behav_df = df[behav]
            # Getting start-stop of each bout
            bouts_df = cls.vect2bouts(behav_df[cls.OutcomesCols.PRED.value] == 1)
            # For each bout (i.e. start-stop pair)
            for _, row in bouts_df.iterrows():
                # Getting only the frames in the current bout
                bout_frames_df = behav_df.loc[row[cls.BoutCols.START.value] : row[cls.BoutCols.STOP.value]]
                # Preparing to make Bout model object
                bout_dict = {
                    cls.BoutCols.START.value: row[cls.BoutCols.START.value],
                    cls.BoutCols.STOP.value: row[cls.BoutCols.STOP.value],
                    cls.BoutCols.DUR.value: row[cls.BoutCols.DUR.value],
                    cls.BoutCols.BEHAV.value: behav,
                    cls.BoutCols.ACTUAL.value: int(mode(bout_frames_df[cls.OutcomesCols.ACTUAL.value]).mode),
                    cls.BoutCols.USER_DEFINED.value: {},
                }
                # Getting the value for the bout (for user_defined behavs only (i.e. columns not in BehavColumns))
                for outcome, values in bout_frames_df.items():
                    if outcome not in enum2tuple(cls.OutcomesCols):
                        # Using mode as proxy for the entire bout's user_defined value
                        bout_dict[cls.BoutCols.USER_DEFINED.value][str(outcome)] = int(mode(values).mode)
                # Making the Bout model object and appending to bouts_ls
                bouts_ls.append(Bout.model_validate(bout_dict))
        # Making and return the Bouts model object
        return Bouts(
            start=df.index[0],
            stop=df.index[-1] + 1,
            bouts=bouts_ls,
            bouts_struct=cls.get_bouts_struct_from_df(df),
        )

    @classmethod
    def bouts2frames(cls, bouts: Bouts) -> pd.DataFrame:
        """
        Bouts model object to frames df.
        """
        # Making dataframe
        df = cls.init_df(pd.Series(np.arange(bouts.start, bouts.stop)))
        # Making columns (for each behaviour, and for pred, actual, and user_defined)
        for bout_struct in bouts.bouts_struct:
            behav = bout_struct.behav
            df[(behav, cls.OutcomesCols.PRED.value)] = 0
            df[(behav, cls.OutcomesCols.ACTUAL.value)] = 0
            for user_defined_i in bout_struct.user_defined:
                df[(behav, user_defined_i)] = 0
        # Sorting columns
        df = df.sort_index(axis=1)
        # Filling in all user_defined columns for each behaviour
        for bout in bouts.bouts:
            bout_ret_df = df.loc[bout.start : bout.stop]
            # Filling in predicted behaviour column
            bout_ret_df.loc[:, (bout.behav, cls.OutcomesCols.PRED.value)] = 1
            # Filling in actual behaviour column
            bout_ret_df.loc[:, (bout.behav, cls.OutcomesCols.ACTUAL.value)] = bout.actual
            # Filling in user_defined columns
            for k, v in bout.user_defined.items():
                bout_ret_df.loc[:, (bout.behav, k)] = v
        # Returning frames df
        return df
