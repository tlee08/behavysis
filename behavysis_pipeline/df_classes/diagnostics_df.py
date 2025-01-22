"""
Utility functions.
"""

import os
from enum import Enum

import pandas as pd
from natsort import natsort_keygen

from behavysis_pipeline.df_classes.df_mixin import DFMixin
from behavysis_pipeline.utils.misc_utils import enum2tuple

####################################################################################################
# DF CONSTANTS
####################################################################################################


class DiagnosticsIN(Enum):
    EXPERIMENT = "experiment"


class DiagnosticsCN(Enum):
    FEATURES = "features"


####################################################################################################
# DF CLASS
####################################################################################################


class DiagnosticsDf(DFMixin):
    """
    Mixin for features DF
    (generated from SimBA feature extraction)
    functions.
    """

    NULLABLE = True
    IN = DiagnosticsIN
    CN = DiagnosticsCN

    @classmethod
    def read(cls, fp: str) -> pd.DataFrame:
        # Reading the file
        df = pd.read_csv(fp, index_col=0)
        # Sorting by index
        df = df.sort_index()
        # Checking after reading
        cls.check_df(df)
        # Returning
        return df

    @classmethod
    def write(cls, df: pd.DataFrame, fp: str) -> None:
        # Checking before writing
        cls.check_df(df)
        # Making the directory if it doesn't exist
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        # Writing the file
        df.to_csv(fp)

    @classmethod
    def init_from_dd_ls(cls, dd_ls: list[dict]) -> pd.DataFrame:
        """
        Initialises the features dataframe from a list of dictionaries.
        """
        # Asserting that dd_ls has the "experiment" key
        assert all("experiment" in dd for dd in dd_ls), "All dictionaries must have the 'experiment' key."
        # Init df
        df = pd.DataFrame(dd_ls).set_index("experiment").sort_index(key=natsort_keygen())
        # Updating index name
        df.index.name = enum2tuple(cls.IN)[0]
        # Updating column names
        df.columns.names = enum2tuple(cls.CN)
        # Checking after init
        cls.check_df(df)
        # Returning
        return df
