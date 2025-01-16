"""
Utility functions.
"""

from __future__ import annotations

import os
from enum import Enum, EnumType

import pandas as pd

from behavysis_pipeline.utils.logging_utils import init_logger
from behavysis_pipeline.utils.misc_utils import MiscMixin

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

    ###############################################################################################
    # DF Read Functions
    ###############################################################################################

    @classmethod
    def read_dlc_csv(cls, fp: str) -> pd.DataFrame:
        """
        Reading DLC dataframe csv file.
        """
        # Reading the file
        df = pd.read_csv(fp, index_col=0)
        # Sorting by index
        df = df.sort_index()
        # Checking after reading
        cls.check_df(df)
        # Returning
        return df

    @classmethod
    def read_h5(cls, fp: str) -> pd.DataFrame:
        """
        Reading dataframe h5 file.
        """
        df = pd.DataFrame(pd.read_hdf(fp, mode="r"))
        # Sorting by index
        df = df.sort_index()
        # Checking after reading
        cls.check_df(df)
        # Returning
        return df

    @classmethod
    def read_feather(cls, fp: str) -> pd.DataFrame:
        """
        Reading dataframe feather file.
        """
        df = pd.read_feather(fp)
        # Sorting by index
        df = df.sort_index()
        # Checking after reading
        cls.check_df(df)
        # Returning
        return df

    @classmethod
    def read_parquet(cls, fp: str) -> pd.DataFrame:
        """
        Reading dataframe parquet file.
        """
        df = pd.read_parquet(fp)
        # Sorting by index
        df = df.sort_index()
        # Checking after reading
        cls.check_df(df)
        # Returning
        return df

    @classmethod
    def read(cls, fp: str) -> pd.DataFrame:
        """
        Default dataframe read method.

        `cls.read_feather`
        """
        return cls.read_feather(fp)

    ###############################################################################################
    # DF Write Functions
    ###############################################################################################

    @classmethod
    def write_dlc_csv(cls, df: pd.DataFrame, fp: str) -> None:
        """
        Writing DLC dataframe to csv file.
        """
        # Checking before writing
        cls.check_df(df)
        # Making the directory if it doesn't exist
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        # Writing the file
        df.to_csv(fp)

    @classmethod
    def write_h5(cls, df: pd.DataFrame, fp: str) -> None:
        """
        Writing dataframe h5 file.
        """
        # Checking before writing
        cls.check_df(df)
        # Making the directory if it doesn't exist
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        # Writing the file
        DLC_HDF_KEY = "data"
        df.to_hdf(fp, key=DLC_HDF_KEY, mode="w")

    @classmethod
    def write_feather(cls, df: pd.Series | pd.DataFrame, fp: str) -> None:
        """
        Writing dataframe feather file.
        """
        # Checking before writing
        cls.check_df(df)
        # Making the directory if it doesn't exist
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        # Writing the file
        df.to_feather(fp)

    @classmethod
    def write_parquet(cls, df: pd.DataFrame, fp: str) -> None:
        """
        Writing dataframe feather file.
        """
        # Checking before writing
        cls.check_df(df)
        # Making the directory if it doesn't exist
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        # Writing the file
        df.to_parquet(fp)

    @classmethod
    def write(cls, df: pd.DataFrame, fp: str) -> None:
        """
        Default dataframe read method.

        `cls.write_feather`
        """
        return cls.write_feather(df, fp)

    ###############################################################################################
    # DF init functions
    ###############################################################################################

    @classmethod
    def init_df(cls, frame_vect: pd.Series | pd.Index) -> pd.DataFrame:
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
        IN = cls.IN or [None]
        CN = cls.CN or [None]
        return pd.DataFrame(
            index=pd.Index(frame_vect, name=MiscMixin.enum2tuple(IN)[0]),
            columns=pd.MultiIndex.from_tuples((), names=MiscMixin.enum2tuple(CN)),
        )

    ###############################################################################################
    # DF Check functions
    ###############################################################################################

    @classmethod
    def check_df(cls, df: pd.DataFrame) -> None:
        """__summary__"""
        # Checking that df is a DataFrame
        assert isinstance(df, pd.DataFrame), "The dataframe is not a pandas DataFrame."
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
            levels = MiscMixin.enum2tuple(levels)
        elif isinstance(levels, str):  # If str
            levels = (levels,)
        assert df.index.names == levels, f"The index level is incorrect. Expected {levels} but got {df.index.names}."

    @classmethod
    def check_CN(cls, df: pd.DataFrame, levels: EnumType | tuple[str] | str) -> None:
        """__summary__"""
        # Converting `levels` to a tuple
        if isinstance(levels, EnumType):  # If Enum
            levels = MiscMixin.enum2tuple(levels)
        elif isinstance(levels, str):  # If str
            levels = (levels,)
        assert (
            df.columns.names == levels
        ), f"The column level is incorrect. Expected {levels} but got {df.columns.names}."
