"""
Utility functions.
"""

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
    FEATURES = "functions"


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
    IO = "csv"

    @classmethod
    def init_from_dd_ls(cls, dd_ls: list[dict]) -> pd.DataFrame:
        """
        Initialises the features dataframe from a list of dictionaries.
        """
        assert all("experiment" in dd for dd in dd_ls), "All dictionaries must have the 'experiment' key."
        df = pd.DataFrame(dd_ls).set_index("experiment").sort_index(key=natsort_keygen())
        df.index.names = list(enum2tuple(cls.IN))
        df.columns.names = list(enum2tuple(cls.CN))
        cls.check_df(df)
        return df
