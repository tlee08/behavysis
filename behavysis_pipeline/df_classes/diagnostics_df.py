"""
Utility functions.
"""

from enum import Enum

import numpy as np
import pandas as pd
from natsort import natsorted

from behavysis_pipeline.df_classes.df_mixin import DFMixin

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
        df = pd.DataFrame(dd_ls).set_index("experiment")
        df = cls.basic_clean(df)
        return df

    @classmethod
    def basic_clean(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = super().basic_clean(df)
        # Natural sort the index
        index = df.index.get_level_values(cls.IN.EXPERIMENT.value)
        assert np.all(np.equal(natsorted(index), natsorted(index.unique())))
        df = df.loc[natsorted(index), :]
        return df
