"""
Functions have the following format:

Parameters
----------
dlc_fp : str
    The DLC dataframe filepath of the experiment to analyse.
ANALYSE_DIR : str
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

import pandas as pd
from behavysis_core.df_classes.analyse_combine_df import AnalyseCombineDf
from behavysis_core.df_classes.analyse_df import AnalyseDf
from behavysis_core.df_classes.df_mixin import DFMixin
from behavysis_core.mixins.io_mixin import IOMixin

###################################################################################################
#               ANALYSIS API FUNCS
###################################################################################################


class AnalyseCombine:
    """__summary__"""

    @staticmethod
    def analyse_combine(
        ANALYSE_DIR: str,
        out_fp: str,
        configs_fp: str,
        # bins: list,
        # summary_func: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> str:
        """
        Takes a behavs dataframe and generates a summary and binned version of the data.
        """
        # TODO: maybe refactor to own folder
        # (because it's another step of the pipeline)
        outcome = ""
        name = IOMixin.get_name(configs_fp)
        # For each analysis subdir, combining fbf files
        analysis_ls = [
            i
            for i in os.listdir(ANALYSE_DIR)
            if os.path.isdir(os.path.join(ANALYSE_DIR, i))
        ]
        # If no analysis files, then return warning and don't make df
        if len(analysis_ls) == 0:
            outcome += "WARNING: no analysis fbf files made. Run `exp.analyse` first"
            return outcome
        # Reading in each fbf analysis df
        comb_df_ls = [
            AnalyseDf.read_feather(
                os.path.join(ANALYSE_DIR, i, "fbf", f"{name}.feather")
            )
            for i in analysis_ls
        ]
        # Making combined df from list of dfs
        comb_df = pd.concat(
            comb_df_ls,
            axis=1,
            keys=analysis_ls,
            names=[AnalyseCombineDf.CN.ANALYSIS.value],
        )
        # Writing to file
        DFMixin.write_feather(comb_df, out_fp)
        # Returning outcome
        return outcome
