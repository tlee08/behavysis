"""
Utility functions.
"""

import os
from enum import Enum, EnumType

import pandas as pd

from behavysis_pipeline.constants import DF_IO_FORMAT
from behavysis_pipeline.utils.logging_utils import init_logger
from behavysis_pipeline.utils.misc_utils import enum2tuple

####################################################################################################
# DF CONSTANTS
####################################################################################################


class FramesIN(Enum):
    FRAME = "frame"


####################################################################################################
# DF CLASS
####################################################################################################


class DFMixin:
    """__summary"""

    logger = init_logger(__name__)

    NULLABLE = True
    IN = None
    CN = None
    IO = DF_IO_FORMAT

    ###############################################################################################
    # DF Read Functions
    ###############################################################################################

    @classmethod
    def read_csv(cls, fp: str) -> pd.DataFrame:
        """Reading DLC dataframe csv file."""
        df = pd.read_csv(
            fp,
            index_col=list(range(len(enum2tuple(cls.IN) if cls.IN else (None,)))),
            header=list(range(len(enum2tuple(cls.CN) if cls.CN else (None,)))),
        )
        df = cls.basic_clean(df)
        return df

    @classmethod
    def read_h5(cls, fp: str) -> pd.DataFrame:
        """Reading dataframe h5 file."""
        df = pd.DataFrame(pd.read_hdf(fp, mode="r"))
        df = cls.basic_clean(df)
        return df

    @classmethod
    def read_feather(cls, fp: str) -> pd.DataFrame:
        """Reading dataframe feather file."""
        df = pd.read_feather(fp)
        df = cls.basic_clean(df)
        return df

    @classmethod
    def read_parquet(cls, fp: str) -> pd.DataFrame:
        """Reading dataframe parquet file."""
        df = pd.read_parquet(fp)
        df = cls.basic_clean(df)
        return df

    @classmethod
    def read(cls, fp: str) -> pd.DataFrame:
        """
        Default dataframe read method.
        Based on `IO` class attribute.
        """
        methods = {
            "csv": cls.read_csv,
            "h5": cls.read_h5,
            "feather": cls.read_feather,
            "parquet": cls.read_parquet,
        }
        assert cls.IO in methods, (
            f"File type, {cls.IO}, not supported.\n" f"Supported IO types are: {list(methods.keys())}."
        )
        return methods[cls.IO](fp)

    ###############################################################################################
    # DF Write Functions
    ###############################################################################################

    @classmethod
    def write_csv(cls, df: pd.DataFrame, fp: str) -> None:
        """Writing DLC dataframe to csv file."""
        df = cls.basic_clean(df)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        df.to_csv(fp)

    @classmethod
    def write_h5(cls, df: pd.DataFrame, fp: str) -> None:
        """Writing dataframe h5 file."""
        df = cls.basic_clean(df)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        df.to_hdf(fp, key="data", mode="w")

    @classmethod
    def write_feather(cls, df: pd.DataFrame, fp: str) -> None:
        """Writing dataframe feather file."""
        df = cls.basic_clean(df)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        df.to_feather(fp)

    @classmethod
    def write_parquet(cls, df: pd.DataFrame, fp: str) -> None:
        """Writing dataframe parquet file."""
        df = cls.basic_clean(df)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        df.to_parquet(fp)

    @classmethod
    def write(cls, df: pd.DataFrame, fp: str) -> None:
        """
        Default dataframe read method based on IO attribute.
        Based on `IO` class attribute.
        """
        methods = {
            "csv": cls.write_csv,
            "h5": cls.write_h5,
            "feather": cls.write_feather,
            "parquet": cls.write_parquet,
        }
        assert cls.IO in methods, (
            f"File type, {cls.IO}, not supported.\n" f"Supported IO types are: {list(methods.keys())}."
        )
        return methods[cls.IO](df, fp)

    ###############################################################################################
    # DF init functions
    ###############################################################################################

    @classmethod
    def init_df(cls, index: pd.Series | pd.Index) -> pd.DataFrame:
        """
        # TODO: write better docstring
        Returning a frame-by-frame analysis_df with the frame number (according to original video)
        as the MultiIndex index, relative to the first element of frame_vect.
        Note that that the frame number can thus begin on a non-zero number.

        Parameters
        ----------
        frame_vect : pd.Series | pd.Index
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        IN = enum2tuple(cls.IN) if cls.IN else (None,)
        CN = enum2tuple(cls.CN) if cls.CN else (None,)
        index_df = index.to_frame()
        return pd.DataFrame(
            index=pd.MultiIndex.from_frame(index_df, names=IN),
            columns=pd.MultiIndex.from_tuples((), names=CN),
        )

    @classmethod
    def basic_clean(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic cleaning of the dataframe. Includes:
        - Setting the index and column names
        - Sorting the index

        Also checks that the df structure is as expected with `check_df`.
        """
        df.index.set_names(enum2tuple(cls.IN) if cls.IN else (None,))
        df.columns.set_names(enum2tuple(cls.CN) if cls.CN else (None,))
        df = df.sort_index()
        df = df.sort_index(axis=1)
        cls.check_df(df)
        return df

    ###############################################################################################
    # DF Check functions
    ###############################################################################################

    @classmethod
    def check_df(cls, df: pd.DataFrame) -> None:
        """__summary__"""
        # Checking that df is a DataFrame
        assert isinstance(df, pd.DataFrame), "The dataframe must be a pandas DataFrame."
        # Checking there are no null values
        if not cls.NULLABLE:
            assert (
                not df.isnull().values.any()
            ), "The dataframe contains null values. Be sure to run interpolate_points first."
        # Checking that the index levels are correct
        if cls.IN:
            cls.check_IN(df, cls.IN)
        # Checking that the column levels are correct
        if cls.CN:
            cls.check_CN(df, cls.CN)

    @classmethod
    def check_IN(cls, df: pd.DataFrame, levels: EnumType | tuple[str] | str) -> None:
        """__summary__"""
        # Converting `levels` to a tuple
        if isinstance(levels, EnumType):  # If Enum
            levels = enum2tuple(levels)
        elif isinstance(levels, str):  # If str
            levels = (levels,)
        assert df.index.names == levels, f"The index level is incorrect. Expected {levels} but got {df.index.names}."

    @classmethod
    def check_CN(cls, df: pd.DataFrame, levels: EnumType | tuple[str] | str) -> None:
        """__summary__"""
        # Converting `levels` to a tuple
        if isinstance(levels, EnumType):  # If Enum
            levels = enum2tuple(levels)
        elif isinstance(levels, str):  # If str
            levels = (levels,)
        assert (
            df.columns.names == levels
        ), f"The column level is incorrect. Expected {levels} but got {df.columns.names}."
