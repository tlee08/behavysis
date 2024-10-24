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

import os
from enum import Enum

import pandas as pd
from behavysis_core.df_mixins.df_io_mixin import DFIOMixin
from behavysis_core.mixins.io_mixin import IOMixin

from behavysis_pipeline.processes.analyse.analyse_mixin import AnalyseMixin

####################################################################################################
# ANALYSIS DATAFRAME CONSTANTS
####################################################################################################


class AnalysisCombineCN(Enum):
    """Enum for the columns in the analysis dataframe."""

    ANALYSIS = "analysis"
    INDIVIDUALS = "individuals"
    MEASURES = "measures"


#####################################################################
#               ANALYSIS API FUNCS
#####################################################################


class AnalyseCombine:
    """__summary__"""

    @staticmethod
    def analyse_combine(
        analysis_dir: str,
        out_dir: str,
        configs_fp: str,
        # bins: list,
        # summary_func: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> str:
        """
        Takes a behavs dataframe and generates a summary and binned version of the data.
        """
        outcome = ""
        name = IOMixin.get_name(configs_fp)
        # For each analysis subdir, combining fbf files
        analysis_ls = [
            i
            for i in os.listdir(analysis_dir)
            if os.path.isdir(os.path.join(analysis_dir, i))
        ]
        # If no analysis files, then return warning and don't make df
        if len(analysis_ls) == 0:
            outcome += "WARNING: no analysis fbf files made. Run `exp.analyse` first"
            return outcome
        comb_df_ls = [
            AnalyseMixin.read_feather(
                os.path.join(analysis_dir, i, "fbf", f"{name}.feather")
            )
            for i in analysis_ls
        ]
        # Making combined df from list of dfs
        comb_df = pd.concat(
            comb_df_ls,
            axis=1,
            keys=analysis_ls,
            names=[AnalysisCombineCN.ANALYSIS.value],
        )
        # Writing to file
        out_fp = os.path.join(out_dir, f"{name}.feather")
        DFIOMixin.write_feather(comb_df, out_fp)
        # Returning outcome
        return outcome
