"""
Utility functions.
"""

from __future__ import annotations

import os
from enum import Enum

import pandas as pd

from behavysis_pipeline.df_classes.df_mixin import DFMixin

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
