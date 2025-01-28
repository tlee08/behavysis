"""
Utility functions.
"""

from enum import Enum

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import mode

from behavysis.df_classes.df_mixin import DFMixin, FramesIN
from behavysis.pydantic_models.bouts import Bout, Bouts, BoutStruct
from behavysis.utils.misc_utils import enum2tuple


class OutcomesPredictedCols(Enum):
    PROB = "prob"
    PRED = "pred"


class OutcomesScoredCols(Enum):
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


class BehavCN(Enum):
    BEHAVS = "behavs"
    OUTCOMES = "outcomes"


class BehavDf(DFMixin):
    NULLABLE = False
    IN = FramesIN
    CN = BehavCN

    OutcomesCols = None

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
            # For each behaviour
            for behav in columns.unique(cls.CN.BEHAVS.value):
                # For each outcome
                for outcome in enum2tuple(cls.OutcomesCols):
                    # Assert the (behav, outcome) column is present
                    assert (behav, outcome) in columns, (
                        f"Expected {outcome} column for {behav}.\nThe columns in the df are: {columns}"
                    )


class BehavPredictedDf(BehavDf):
    OutcomesCols = OutcomesPredictedCols

    @classmethod
    def check_outcomes_cols(cls, df: pd.DataFrame) -> None:
        # Running regular outcomes columns check
        super().check_outcomes_cols(df)
        # Checking that these are the ONLY outcomes columns
        expected_outcomes_cols = enum2tuple(cls.OutcomesCols)
        df_outcomes_cols = df.columns.unique(cls.CN.OUTCOMES.value)
        assert set(df_outcomes_cols) == set(expected_outcomes_cols), (
            f"Expected ONLY {expected_outcomes_cols} outcomes columns.\n"
            f"The outcomes columns in the df are: {df_outcomes_cols}"
        )


class BehavScoredDf(BehavDf):
    OutcomesCols = OutcomesScoredCols

    @classmethod
    def check_outcomes_cols(cls, df: pd.DataFrame) -> None:
        # Running regular outcomes columns check
        super().check_outcomes_cols(df)
        # Checking that the "PROB" column is not in the outcomes columns
        exclude_outcomes_cols = [BehavPredictedDf.OutcomesCols.PROB.value]
        df_outcome_cols = df.columns.unique(cls.CN.OUTCOMES.value)
        assert not (set(df_outcome_cols) & set(exclude_outcomes_cols)), (
            f"Expected NOT to find {exclude_outcomes_cols} in outcomes columns.\n"
            f"The outcomes columns in the df are: {df_outcome_cols}"
        )

    ###############################################################################################
    # BORIS IMPORT METHODS
    ###############################################################################################

    @classmethod
    def import_boris_tsv(cls, fp: str, behavs_ls: list[str], start_frame: int, stop_frame: int) -> pd.DataFrame:
        """
        Importing Boris TSV file.
        """
        # Making df structure
        df = cls.init_df(pd.Series(np.arange(start_frame, stop_frame)))
        # Reading in corresponding BORIS tsv file
        df_boris = pd.read_csv(fp, sep="\t")
        # Initialising new behaviour columns
        # `behavs_ls` can only include behaviours in BORIS df
        assert np.isin(behavs_ls, df_boris["Behavior"].unique()), (
            f"Only behaviour names in BORIS dataframe are valid.\n"
            f"Given behaviours: {behavs_ls}\n"
            f"BORIS behaviours: {df_boris['Behavior'].unique()}"
        )
        for behav in behavs_ls:
            df[(behav, cls.OutcomesCols.ACTUAL.value)] = BehavValues.NON_BEHAV.value
            df[(behav, cls.OutcomesCols.PRED.value)] = BehavValues.NON_BEHAV.value
        # Setting the classification values from the BORIS file
        for ind, row in df_boris.iterrows():
            # Getting corresponding frame of this event point
            behav = row["Behavior"]
            frame = row["Image index"]
            status = row["Behavior type"]
            # Only using in behav is in behavs_ls
            if behav not in behavs_ls:
                continue
            # Status is either "START" (of behaviour) or "STOP"
            val = BehavValues.BEHAV.value if status == "START" else BehavValues.NON_BEHAV.value
            # Updating the classification in the scored df
            df.loc[frame:, (behav, cls.OutcomesCols.ACTUAL.value)] = val
            df.loc[frame:, (behav, cls.OutcomesCols.PRED.value)] = val
        return df

    @classmethod
    def update_behav(cls, df: pd.DataFrame, old_behav: str, new_behav: str) -> pd.DataFrame:
        """
        Update the given behaviour name (old_behav) with a new name (new_behav).
        """
        # Getting columns
        columns = df.columns.to_frame(index=False)
        # Updating the behaviour column
        columns[cls.CN.BEHAVS.value] = columns[cls.CN.BEHAVS.value].replace(old_behav, new_behav)
        # Setting the new columns
        df.columns = pd.MultiIndex.from_frame(columns)
        return df

    ###############################################################################################
    # CONVERT FROM PREDICTED TO SCORED BEHAV DF
    ###############################################################################################

    @classmethod
    def get_bouts_struct_from_df(cls, df: pd.DataFrame) -> list[BoutStruct]:
        """
        Returns the list BoutStruct objects from the given BehavDf's columns.
        """
        bouts_struct = []
        for behav in df.columns.unique(cls.CN.BEHAVS.value):
            bouts_struct.append(
                BoutStruct(
                    behav=behav,
                    user_defined=[
                        i
                        for i in df[behav].columns.unique(cls.CN.OUTCOMES.value)
                        if i not in enum2tuple(cls.OutcomesCols)
                    ],
                )
            )
        return bouts_struct

    @classmethod
    def predicted2scored(cls, df: pd.DataFrame, bouts_struct: list[BoutStruct] | None = None) -> pd.DataFrame:
        """
        Convert a predicted behaviours dataframe to a scored behaviours dataframe.
        """
        # TODO: bouts struct need to include user_defined as well
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
            # Adding actual column and behav is set to undetermined
            scored_df[(behav, cls.OutcomesCols.ACTUAL.value)] = scored_df[(behav, cls.OutcomesCols.PRED.value)].replace(
                BehavValues.BEHAV.value, BehavValues.UNDETERMINED.value
            )
            # Adding user_defined columns and setting values to 0
            for user_defined_i in user_defined:
                scored_df[(behav, user_defined_i)] = BehavValues.NON_BEHAV.value
        scored_df = cls.basic_clean(scored_df)
        return scored_df

    ###############################################################################################
    # BOUTS FUNCS
    ###############################################################################################

    @classmethod
    def vect2bouts_df(
        cls,
        vect: pd.Series,
    ) -> pd.DataFrame:
        """
        Will return a dataframe with the start and stop indexes of each contiguous set of
        positive/true values (i.e. a bout).

        Parameters
        ----------
        vect : pd.Series
            Expects a vector of booleans

        Returns
        -------
        pd.DataFrame
            _description_
        """
        offset = 0
        if vect.shape[0] > 0:
            # Gets offset from first index
            # NOTE: Also safe for multi-index. Assumes using "frame" level
            offset = vect.index.get_level_values(cls.IN.FRAME.value)[0]
        # Getting stop and start indexes of each bout
        z = np.concatenate(([0], vect, [0]))
        start = np.flatnonzero(~z[:-1] & z[1:])
        stop = np.flatnonzero(z[:-1] & ~z[1:]) - 1
        # Making dataframe
        bouts_df = pd.DataFrame({BoutCols.START.value: start, BoutCols.STOP.value: stop}) + offset
        bouts_df[BoutCols.DUR.value] = bouts_df[BoutCols.STOP.value] - bouts_df[BoutCols.START.value] + 1
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
            bouts_df = cls.vect2bouts_df(behav_df[cls.OutcomesCols.PRED.value] == BehavValues.BEHAV.value)
            # For each bout (i.e. start-stop pair)
            for _, row in bouts_df.iterrows():
                # Getting only the frames in the current bout
                bout_frames_df = behav_df.loc[row[BoutCols.START.value] : row[BoutCols.STOP.value]]
                # Making bout object
                bout = Bout(
                    start=row[BoutCols.START.value],
                    stop=row[BoutCols.STOP.value],
                    dur=row[BoutCols.DUR.value],
                    behav=behav,
                    actual=int(mode(bout_frames_df[cls.OutcomesCols.ACTUAL.value]).mode),
                    user_defined={},
                )
                # Storing user_defined column names in bout object
                for outcome, values in bout_frames_df.items():
                    if outcome not in enum2tuple(cls.OutcomesCols):
                        # Using mode as proxy for the entire bout's user_defined value
                        bout.user_defined[str(outcome)] = int(mode(values).mode)
                bouts_ls.append(bout)
        return Bouts(
            start=df.index.get_level_values(cls.IN.FRAME.value)[0],
            stop=df.index.get_level_values(cls.IN.FRAME.value)[-1] + 1,
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
            df[(behav, cls.OutcomesCols.PRED.value)] = BehavValues.NON_BEHAV.value
            df[(behav, cls.OutcomesCols.ACTUAL.value)] = BehavValues.NON_BEHAV.value
            for user_defined_i in bout_struct.user_defined:
                df[(behav, user_defined_i)] = BehavValues.NON_BEHAV.value
        # Filling in all user_defined columns for each behaviour
        for bout in bouts.bouts:
            df.loc[bout.start : bout.stop, (bout.behav, cls.OutcomesCols.PRED.value)] = BehavValues.BEHAV.value
            df.loc[bout.start : bout.stop, (bout.behav, cls.OutcomesCols.ACTUAL.value)] = bout.actual
            # Filling in user_defined columns
            for k, v in bout.user_defined.items():
                df.loc[bout.start : bout.stop, (bout.behav, k)] = v
        df = cls.basic_clean(df)
        return df

    @classmethod
    def fps_scale_df(cls, df: pd.DataFrame, src_fps: int, dst_fps: int) -> pd.DataFrame:
        fps_scale = dst_fps / src_fps
        df = cls.basic_clean(df)
        columns = df.columns
        index = df.index
        # Scaling the df
        df = np.ceil(ndimage.zoom(df, (fps_scale, 1))).astype(int)
        index = np.round(ndimage.zoom(index, fps_scale)).astype(int)
        # Making new df
        df = pd.DataFrame(df, index=index, columns=columns)
        df = cls.basic_clean(df)
        return df


if __name__ == "__main__":
    # Making test df
    v = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1])
    df0 = BehavScoredDf.init_df(pd.Series(np.arange(v.shape[0])))
    df0[("behav", OutcomesScoredCols.PRED.value)] = v
    df0[("behav", OutcomesScoredCols.ACTUAL.value)] = v
    df0 = BehavScoredDf.basic_clean(df0)
    # Round 1
    b1 = BehavScoredDf.frames2bouts(df0)
    df1 = BehavScoredDf.bouts2frames(b1)
    # Round 2
    b2 = BehavScoredDf.frames2bouts(df1)
    df2 = BehavScoredDf.bouts2frames(b2)
    # Asserting that the dfs and bouts are the same
    assert df1.equals(df2)
    assert b1 == b2
