"""
Utility functions.
"""

from enum import Enum

import numpy as np
import pandas as pd
from natsort import natsorted

from behavysis.df_classes.df_mixin import DFMixin


class DiagnosticsIN(Enum):
    EXPERIMENT = "experiment"


class DiagnosticsCN(Enum):
    FEATURES = "functions"


class DiagnosticsDf(DFMixin):
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
        index = natsorted(df.index.get_level_values(cls.IN.EXPERIMENT.value))
        assert set(index) == set(np.unique(index)), (
            "All experiments must be unique.\n" f"Some duplicates found in the following list of experiments: {index}"
        )
        df = df.loc[index, :]
        return df
