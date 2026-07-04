"""Behavior DataFrame for behavioral classification data."""

from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import mode

from behavysis.df_classes.keypoints_df import FramesIN
from behavysis.models.bouts import Bout, Bouts, BoutStruct
from behavysis.utils.df_mixin import DFMixin


class OutcomesPredictedCols(Enum):
    """OutcomesPredictedCols."""

    PROB = "prob"
    PRED = "pred"


class OutcomesScoredCols(Enum):
    """OutcomesScoredCols."""

    ACTUAL = "actual"


class BehavValues(Enum):
    """BehavValues."""

    TRUE_POS = 1
    TRUE_NEG = 0
    FALSE_POS = -1
    UNSURE = -2


class BoutCols(Enum):
    """BoutCols."""

    START = "start"
    STOP = "stop"
    DUR = "dur"


class BehavCN(Enum):
    """BehavCN."""

    BEHAVS = "behavs"
    OUTCOMES = "outcomes"


class BehavDf(DFMixin):
    """BehavDf."""

    NULLABLE = False
    IN = FramesIN
    CN = BehavCN
    OutcomesCols = None

    @classmethod
    def _validate(cls, df: pd.DataFrame) -> None:
        super()._validate(df)
        if cls.OutcomesCols:
            for behav in df.columns.unique(cls.CN.BEHAVS.value):
                for outcome in (e.value for e in cls.OutcomesCols):
                    assert (behav, outcome) in df.columns, (
                        f"Expected '{outcome}' column for '{behav}'.\n"
                        f"Columns: {df.columns.to_list()}"
                    )


class BehavPredictedDf(BehavDf):
    """BehavPredictedDf."""

    OutcomesCols = OutcomesPredictedCols

    @classmethod
    def _validate(cls, df: pd.DataFrame) -> None:
        super()._validate(df)
        actual_outcomes = set(df.columns.unique(cls.CN.OUTCOMES.value))
        expected = {e.value for e in cls.OutcomesCols}
        assert actual_outcomes == expected, (
            f"Expected ONLY {expected} outcomes columns.\nFound: {actual_outcomes}"
        )


class BehavScoredDf(BehavDf):
    """BehavScoredDf."""

    OutcomesCols = OutcomesScoredCols

    @classmethod
    def read(cls, fp: Path, fmt: str | None = None) -> pd.DataFrame:
        """Read dataframe from file, validate schema, and sort."""
        df = pd.read_parquet(fp)

        # Set index
        df = df.set_index([i.value for i in cls.IN])
        # Set MultiIndex columns
        df.columns = df.columns.str.split("__", expand=True)

        df = cls.clean_and_validate(df)
        for col in df.columns:
            df[col] = df[col].astype(np.int8)
        return df

    @classmethod
    def write(cls, df: pd.DataFrame, fp: Path, fmt: str | None = None) -> None:
        """Validate schema, transform to TS viewer format, and write dataframe to file."""
        df = cls.clean_and_validate(df)
        fp.parent.mkdir(parents=True, exist_ok=True)

        # Transform to TS viewer format: frame as column, __-delimited behaviour columns
        df_out = df.reset_index()
        df_out.columns = ["__".join(filter(None, c)) for c in df_out.columns]
        df_out["frame"] = df_out["frame"].astype(np.int64)
        for col in df_out.columns:
            if col != "frame":
                df_out[col] = df_out[col].astype(np.int8)

        fmt = fmt or cls.IO
        raise_msg = f"Unsupported format: {fmt}. Use: csv, h5, feather, parquet."
        if fmt == "csv":
            df_out.to_csv(fp)
        elif fmt == "h5":
            df_out.to_hdf(fp, key="data", mode="w")
        elif fmt == "feather":
            df_out.to_feather(fp)
        elif fmt == "parquet":
            df_out.to_parquet(fp)
        else:
            raise ValueError(raise_msg)

    @classmethod
    def _validate(cls, df: pd.DataFrame) -> None:
        super()._validate(df)
        exclude = {OutcomesPredictedCols.PROB.value}
        actual = set(df.columns.unique(cls.CN.OUTCOMES.value))
        assert not (actual & exclude), (
            f"Expected NOT to find {exclude} in outcomes.\nFound: {actual}"
        )

    @classmethod
    def import_boris_tsv(
        cls,
        fp: Path,
        behavs_ls: list[str],
        start_frame: int,
        stop_frame: int,
    ) -> pd.DataFrame:
        """Import Boris TSV file to scored DataFrame."""
        df = cls.init_df(pd.Series(np.arange(start_frame, stop_frame)))
        df_boris = pd.read_csv(fp, sep="\t")
        assert np.isin(behavs_ls, df_boris["Behavior"].unique()).all(), (
            f"Some behaviors not in BORIS file.\n"
            f"Requested: {behavs_ls}\n"
            f"BORIS: {df_boris['Behavior'].unique()}"
        )
        for behav in behavs_ls:
            df[(behav, cls.OutcomesCols.ACTUAL.value)] = BehavValues.TRUE_NEG.value
            df[(behav, OutcomesPredictedCols.PRED.value)] = BehavValues.TRUE_NEG.value
        for _, row in df_boris.iterrows():
            behav = row["Behavior"]
            frame = row["Image index"]
            status = row["Behavior type"]
            if behav not in behavs_ls:
                continue
            val = (
                BehavValues.TRUE_POS.value
                if status == "START"
                else BehavValues.TRUE_NEG.value
            )
            df.loc[frame:, (behav, cls.OutcomesCols.ACTUAL.value)] = val
            df.loc[frame:, (behav, OutcomesPredictedCols.PRED.value)] = val
        return cls.clean_and_validate(df)

    @classmethod
    def update_behav(cls, df: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
        """Rename a behavior column."""
        columns = df.columns.to_frame(index=False)
        columns[cls.CN.BEHAVS.value] = columns[cls.CN.BEHAVS.value].replace(old, new)
        df.columns = pd.MultiIndex.from_frame(columns)
        return cls.clean_and_validate(df)

    @classmethod
    def get_bouts_struct(cls, df: pd.DataFrame) -> list[BoutStruct]:
        """Extract BoutStruct from DataFrame columns."""
        standard = {e.value for e in OutcomesScoredCols} | {
            e.value for e in OutcomesPredictedCols
        }
        bouts_struct = []
        for behav in df.columns.unique(cls.CN.BEHAVS.value):
            user_defined = [
                c
                for c in df[behav].columns.unique(cls.CN.OUTCOMES.value)
                if c not in standard
            ]
            bouts_struct.append(BoutStruct(behav=behav, user_defined=user_defined))
        return bouts_struct

    @classmethod
    def predicted2scored(
        cls, df: pd.DataFrame, bouts_struct: list[BoutStruct] | None = None
    ) -> pd.DataFrame:
        """Convert predicted DataFrame to scored DataFrame."""
        bouts_struct = bouts_struct or cls.get_bouts_struct(df)
        scored_df = cls.init_df(df.index)
        for bout in bouts_struct:
            behav = bout.behav
            scored_df[(behav, cls.OutcomesCols.ACTUAL.value)] = (
                df[(behav, OutcomesPredictedCols.PRED.value)]
                .replace(BehavValues.TRUE_POS.value, BehavValues.UNSURE.value)
                .astype(np.int8)
            )
            for user_col in bout.user_defined:
                scored_df[(behav, user_col)] = np.int8(BehavValues.TRUE_NEG.value)
        return cls.clean_and_validate(scored_df)

    @classmethod
    def vect2bouts_df(cls, vect: pd.Series) -> pd.DataFrame:
        """Convert boolean vector to bouts DataFrame with start/stop/dur."""
        offset = 0
        if len(vect) > 0:
            offset = vect.index.get_level_values(cls.IN.FRAME.value)[0]
        z = np.concatenate(([0], vect, [0]))
        start = np.flatnonzero(~z[:-1] & z[1:])
        stop = np.flatnonzero(z[:-1] & ~z[1:]) - 1
        bouts_df = (
            pd.DataFrame({BoutCols.START.value: start, BoutCols.STOP.value: stop})
            + offset
        )
        bouts_df[BoutCols.DUR.value] = (
            bouts_df[BoutCols.STOP.value] - bouts_df[BoutCols.START.value] + 1
        )
        return bouts_df

    @classmethod
    def frames2bouts(cls, df: pd.DataFrame) -> Bouts:
        """Convert frame-level DataFrame to Bouts model."""
        bouts_ls = []
        for behav in df.columns.unique(cls.CN.BEHAVS.value):
            behav_df = df[behav]
            bouts_df = cls.vect2bouts_df(
                behav_df[OutcomesPredictedCols.PRED.value] == BehavValues.TRUE_POS.value
            )
            for _, row in bouts_df.iterrows():
                bout_frames = behav_df.loc[
                    row[BoutCols.START.value] : row[BoutCols.STOP.value]
                ]
                bout = Bout(
                    start=row[BoutCols.START.value],
                    stop=row[BoutCols.STOP.value],
                    dur=row[BoutCols.DUR.value],
                    behav=behav,
                    actual=int(mode(bout_frames[cls.OutcomesCols.ACTUAL.value]).mode),
                    user_defined={},
                )
                for outcome, values in bout_frames.items():
                    if outcome not in (e.value for e in cls.OutcomesCols):
                        bout.user_defined[str(outcome)] = int(mode(values).mode)
                bouts_ls.append(bout)
        return Bouts(
            start=df.index.get_level_values(cls.IN.FRAME.value)[0],
            stop=df.index.get_level_values(cls.IN.FRAME.value)[-1] + 1,
            bouts=bouts_ls,
            bouts_struct=cls.get_bouts_struct(df),
        )

    @classmethod
    def bouts2frames(cls, bouts: Bouts) -> pd.DataFrame:
        """Convert Bouts model to frame-level DataFrame."""
        df = cls.init_df(pd.Series(np.arange(bouts.start, bouts.stop)))
        for bout_struct in bouts.bouts_struct:
            behav = bout_struct.behav
            df[(behav, OutcomesPredictedCols.PRED.value)] = BehavValues.TRUE_NEG.value
            df[(behav, cls.OutcomesCols.ACTUAL.value)] = BehavValues.TRUE_NEG.value
            for user_col in bout_struct.user_defined:
                df[(behav, user_col)] = BehavValues.TRUE_NEG.value
        for bout in bouts.bouts:
            df.loc[
                bout.start : bout.stop,
                (bout.behav, OutcomesPredictedCols.PRED.value),
            ] = BehavValues.TRUE_POS.value
            df.loc[
                bout.start : bout.stop, (bout.behav, cls.OutcomesCols.ACTUAL.value)
            ] = bout.actual
            for k, v in bout.user_defined.items():
                df.loc[bout.start : bout.stop, (bout.behav, k)] = v
        return cls.clean_and_validate(df)

    @classmethod
    def fps_scale(
        cls, df: pd.DataFrame, src_fps: float, dst_fps: float
    ) -> pd.DataFrame:
        """Resample DataFrame to a different frame rate."""
        fps_scale = dst_fps / src_fps
        df = cls.clean_and_validate(df)
        columns = df.columns
        index = df.index
        scaled_vals = np.ceil(ndimage.zoom(df, (fps_scale, 1))).astype(int)
        index_scaled = np.round(ndimage.zoom(index, fps_scale) * fps_scale).astype(int)
        scaled_df = pd.DataFrame(scaled_vals, index=index_scaled, columns=columns)
        return cls.clean_and_validate(scaled_df)


if __name__ == "__main__":
    v = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1])
    df0 = BehavScoredDf.init_df(pd.Series(np.arange(len(v))))
    df0[("behav", OutcomesPredictedCols.PRED.value)] = v
    df0[("behav", OutcomesScoredCols.ACTUAL.value)] = v
    df0 = BehavScoredDf.clean_and_validate(df0)
    b1 = BehavScoredDf.frames2bouts(df0)
    df1 = BehavScoredDf.bouts2frames(b1)
    b2 = BehavScoredDf.frames2bouts(df1)
    df2 = BehavScoredDf.bouts2frames(b2)
    assert df1.equals(df2), "DataFrames should be equal"
    assert b1 == b2, "Bouts should be equal"
