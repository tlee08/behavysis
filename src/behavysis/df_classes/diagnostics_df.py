"""Diagnostics DataFrame for tracking pipeline processing results."""

from enum import Enum

import pandas as pd
from natsort import natsorted

from behavysis.utils.df_mixin import DFMixin


class DiagnosticsIN(Enum):
    EXPERIMENT = "experiment"


class DiagnosticsCN(Enum):
    FUNCTIONS = "functions"


class DiagnosticsDf(DFMixin):
    NULLABLE = True
    IN = DiagnosticsIN
    CN = DiagnosticsCN
    IO = "csv"

    @classmethod
    def init_from_results(cls, results: list[dict]) -> pd.DataFrame:
        """Create DataFrame from list of result dictionaries."""
        assert all("experiment" in r for r in results), "Missing 'experiment' key"
        df = pd.DataFrame(results).set_index("experiment")
        return cls._clean_and_validate(df)

    @classmethod
    def _clean_and_validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = super()._clean_and_validate(df)
        # Natural sort index
        index = natsorted(df.index.get_level_values(cls.IN.EXPERIMENT.value))
        assert len(index) == len(set(index)), f"Duplicate experiments found: {index}"
        return df.loc[index, :]
