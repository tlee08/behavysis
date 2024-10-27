"""
Functions have the following format:

Parameters
----------
dlc_fp : str
    The DLC dataframe filepath of the experiment to analyse.
analysis_dir : str
    The analysis directory path.
configs_fp : str
    the experiment's JSON configs file.

Returns
-------
str
    The outcome of the process.
"""

from __future__ import annotations

from enum import Enum

import pandas as pd
from behavysis_core.constants import FramesIN

####################################################################################################
# ANALYSIS DATAFRAME CONSTANTS
####################################################################################################


class AnalyseCombineCN(Enum):
    """Enum for the columns in the analysis dataframe."""

    ANALYSIS = "analysis"
    INDIVIDUALS = "individuals"
    MEASURES = "measures"


#####################################################################
#               ANALYSIS API FUNCS
#####################################################################


class AnalyseCombineDfMixin:
    """__summary__"""

    @staticmethod
    def init_df(frame_vect: pd.Series | pd.Index) -> pd.DataFrame:
        """
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
        return pd.DataFrame(
            index=pd.Index(frame_vect, name=DFMixin.enum2tuple(FramesIN)[0]),
            columns=pd.MultiIndex.from_tuples(
                (), names=DFMixin.enum2tuple(AnalyseCombineCN)
            ),
        )

    @staticmethod
    def check_df(df: pd.DataFrame) -> None:
        """
        Checks whether the dataframe is in the correct format for the keypoints functions.

        Checks that:

        - There are no null values.
        - The column levels are correct.
        - The index levels are correct.
        """
        print("ABCD")
        print(df)
        print(df.columns.names)
        # Checking for null values
        assert not df.isnull().values.any(), "The dataframe contains null values. Be sure to run interpolate_points first."
        # Checking that the index levels are correct
        DFMixin.check_df_index_names(df, DFMixin.enum2tuple(FramesIN))
        # Checking that the column levels are correct
        DFMixin.check_df_column_names(df, DFMixin.enum2tuple(AnalyseCombineCN))

    @staticmethod
    def read_feather(fp: str) -> pd.DataFrame:
        """
        Reading feather file.
        """
        # Reading
        df = DFMixin.read_feather(fp)
        # Checking
        AnalyseCombineDfMixin.check_df(df)
        # Returning
        return df
