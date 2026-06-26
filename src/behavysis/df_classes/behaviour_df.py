"""Behaviour DataFrame for behavioural classification data."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mode

from behavysis.constants import (
    ACTUAL,
    BEHAVIOUR,
    DUR,
    FRAME,
    OUTCOMES,
    PRED,
    PROB,
    START,
    STOP,
    TRUE_NEG,
    TRUE_POS,
    UNSURE,
)
from behavysis.models import Bout, Bouts, BoutStruct

from .df_mixin import DFMixin

# TODO: depend on config, rather than inference to build BoutStruct
# Consider making vect2bouts and predicted2scored into standalone functions


class BehaviourPredictedDf(DFMixin):
    """BehavPredictedDf."""

    is_nullable = False
    index_names = (FRAME,)
    column_names = (BEHAVIOUR, OUTCOMES)

    @classmethod
    def _validate(cls, df: pd.DataFrame) -> None:
        super()._validate(df)
        expected = {PROB, PRED}
        actual = set(df.columns.unique(OUTCOMES))
        assert actual == expected, (
            f"Expected ONLY {expected} outcomes columns.\nFound: {actual}"
        )


class BehaviourScoredDf(DFMixin):
    """BehaviourScoredDf."""

    is_nullable = False
    index_names = (FRAME,)
    column_names = (BEHAVIOUR, OUTCOMES)

    @classmethod
    def _validate(cls, df: pd.DataFrame) -> None:
        super()._validate(df)
        expected = {PRED, ACTUAL}
        actual = set(df.columns.unique(OUTCOMES))
        assert actual.issubset(expected), (
            f"Expected to include subset {expected} outcomes columns.\nFound: {actual}"
        )

    @classmethod
    def import_boris_tsv(
        cls,
        fp: Path,
        behaviour_ls: list[str],
        start_frame: int,
        stop_frame: int,
    ) -> pd.DataFrame:
        """Import Boris TSV file to scored DataFrame."""
        df = cls.init_df(pd.Series(np.arange(start_frame, stop_frame)))
        df_boris = pd.read_csv(fp, sep="\t")
        assert np.isin(behaviour_ls, df_boris["Behaviour"].unique()).all(), (
            f"Some behaviours not in BORIS file.\n"
            f"Requested: {behaviour_ls}\n"
            f"BORIS: {df_boris['Behaviour'].unique()}"
        )
        for behav in behaviour_ls:
            df[(behav, ACTUAL)] = TRUE_NEG
            df[(behav, PRED)] = TRUE_NEG
        for _, row in df_boris.iterrows():
            behav = row["Behaviour"]
            frame = row["Image index"]
            status = row["Behaviour type"]
            if behav not in behaviour_ls:
                continue
            val = TRUE_POS if status == "START" else TRUE_NEG
            df.loc[frame:, (behav, ACTUAL)] = val
            df.loc[frame:, (behav, PRED)] = val
        return cls.clean_and_validate(df)

    @classmethod
    def get_bouts_struct(cls, df: pd.DataFrame) -> list[BoutStruct]:
        """Extract BoutStruct from DataFrame columns."""
        bouts_struct = []
        for behav in df.columns.unique(BEHAVIOUR):
            user_defined = [
                c for c in df[behav].columns.unique(OUTCOMES) if c not in [PRED, ACTUAL]
            ]
            bouts_struct.append(BoutStruct(behav=behav, user_defined=user_defined))
        return bouts_struct

    @classmethod
    def predicted2scored(
        cls,
        df: pd.DataFrame,
        bouts_struct: list[BoutStruct],
    ) -> pd.DataFrame:
        """Convert predicted DataFrame to scored DataFrame."""
        scored_df = cls.init_df(df.index)
        for bout in bouts_struct:
            behav = bout.behav
            scored_df[(behav, PRED)] = df[(behav, PRED)].to_numpy()
            scored_df[(behav, ACTUAL)] = scored_df[(behav, PRED)].replace(
                TRUE_POS,
                UNSURE,
            )
            for user_col in bout.user_defined:
                scored_df[(behav, user_col)] = TRUE_NEG
        return cls.clean_and_validate(scored_df)

    @classmethod
    def vect2bouts_df(cls, vect: pd.Series) -> pd.DataFrame:
        """Convert boolean vector to bouts DataFrame with start/stop/dur."""
        offset = 0
        if len(vect) > 0:
            offset = vect.index.get_level_values(FRAME)[0]
        z = np.concatenate(([0], vect, [0]))
        start = np.flatnonzero(~z[:-1] & z[1:])
        stop = np.flatnonzero(z[:-1] & ~z[1:]) - 1
        bouts_df = pd.DataFrame({START: start, STOP: stop}) + offset
        bouts_df[DUR] = bouts_df[STOP] - bouts_df[START] + 1
        return bouts_df

    @classmethod
    def frames2bouts(cls, df: pd.DataFrame) -> Bouts:
        """Convert frame-level DataFrame to Bouts model."""
        bouts_ls = []
        for behav in df.columns.unique(BEHAVIOUR):
            behav_df = df[behav]
            bouts_df = cls.vect2bouts_df(behav_df[PRED] == TRUE_POS)
            for _, row in bouts_df.iterrows():
                bout_frames = behav_df.loc[row[START] : row[STOP]]
                bout = Bout(
                    start=row[START],
                    stop=row[STOP],
                    dur=row[DUR],
                    behav=behav,
                    actual=int(mode(bout_frames[ACTUAL]).mode),
                    user_defined={},
                )
                for outcome, values in bout_frames.items():
                    if outcome not in [PRED, ACTUAL]:
                        bout.user_defined[str(outcome)] = int(mode(values).mode)
                bouts_ls.append(bout)
        return Bouts(
            start=df.index.get_level_values(FRAME)[0],
            stop=df.index.get_level_values(FRAME)[-1] + 1,
            bouts=bouts_ls,
            bouts_struct=cls.get_bouts_struct(df),
        )

    @classmethod
    def bouts2frames(cls, bouts: Bouts) -> pd.DataFrame:
        """Convert Bouts model to frame-level DataFrame."""
        df = cls.init_df(pd.Series(np.arange(bouts.start, bouts.stop)))
        for bout_struct in bouts.bouts_struct:
            behav = bout_struct.behav
            df[(behav, PRED)] = TRUE_NEG
            df[(behav, ACTUAL)] = TRUE_NEG
            for user_col in bout_struct.user_defined:
                df[(behav, user_col)] = TRUE_NEG
        for bout in bouts.bouts:
            df.loc[bout.start : bout.stop, (bout.behav, PRED)] = TRUE_POS
            df.loc[bout.start : bout.stop, (bout.behav, ACTUAL)] = bout.actual
            for k, v in bout.user_defined.items():
                df.loc[bout.start : bout.stop, (bout.behav, k)] = v
        return cls.clean_and_validate(df)


if __name__ == "__main__":
    v = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1])
    df0 = BehaviourScoredDf.init_df(pd.Series(np.arange(len(v))))
    df0[(BEHAVIOUR, PRED)] = v
    df0[(BEHAVIOUR, ACTUAL)] = v
    df0 = BehaviourScoredDf.clean_and_validate(df0)
    b1 = BehaviourScoredDf.frames2bouts(df0)
    df1 = BehaviourScoredDf.bouts2frames(b1)
    b2 = BehaviourScoredDf.frames2bouts(df1)
    df2 = BehaviourScoredDf.bouts2frames(b2)
    assert df1.equals(df2), "DataFrames should be equal"
    assert b1 == b2, "Bouts should be equal"
