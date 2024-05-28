import os
import pandas as pd
from typing import Callable
import functools

from behavysis_core.mixins.df_io_mixin import DFIOMixin
from behavysis_core.mixins.diagnostics_mixin import DiagnosticsMixin


class Export:
    """__summary__"""

    @staticmethod
    def export_decorator(
        func: Callable[[str, str, bool], str]
    ) -> Callable[[str, str, bool], str]:
        """__summary__"""

        @functools.wraps(func)
        def wrapper(in_fp: str, out_fp: str, overwrite: bool) -> str:
            # If overwrite is False, checking if we should skip processing
            if not overwrite and os.path.exists(out_fp):
                return DiagnosticsMixin.warning_msg()
            # Running the function
            res = func(in_fp, out_fp, overwrite)
            # Returning outcome
            return f"Exported dataframe from {in_fp} to {out_fp}.\n" + res

        return wrapper

    @staticmethod
    @export_decorator
    def feather_2_feather(in_fp: str, out_fp: str, overwrite: bool) -> str:
        """__summary__"""
        # Reading file
        df = DFIOMixin.read_feather(in_fp)
        # Writing file
        DFIOMixin.write_feather(df, out_fp)
        # Returning outcome
        return "feather to feather\n"

    @staticmethod
    def feather_2_csv(in_fp: str, out_fp: str, overwrite: bool) -> str:
        """__summary__"""
        # Reading file
        df = DFIOMixin.read_feather(in_fp)
        # Writing file
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        df.to_csv(out_fp)
        # Returning outcome
        return "feather to csv\n"
