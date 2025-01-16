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

from behavysis_pipeline.df_classes.analyse_combined_df import AnalyseCombinedDf
from behavysis_pipeline.df_classes.analyse_df import AnalyseDf
from behavysis_pipeline.utils.diagnostics_utils import file_exists_msg
from behavysis_pipeline.utils.io_utils import get_name
from behavysis_pipeline.utils.logging_utils import init_logger, logger_func_decorator

###################################################################################################
#               ANALYSIS API FUNCS
###################################################################################################


class AnalyseCombine:
    """__summary__"""

    logger = init_logger(__name__)

    @classmethod
    @logger_func_decorator(logger)
    def analyse_combine(
        analyse_dir: str,
        out_fp: str,
        configs_fp: str,
        # bins: list,
        # summary_func: Callable[[pd.DataFrame], pd.DataFrame],
        overwrite: bool,
    ) -> str:
        """
        Takes a behavs dataframe and generates a summary and binned version of the data.
        """
        if not overwrite and os.path.exists(out_fp):
            return file_exists_msg(out_fp)
        outcome = ""
        name = get_name(configs_fp)
        # For each analysis subdir, combining fbf files
        analysis_subdir_ls = [i for i in os.listdir(analyse_dir) if os.path.isdir(os.path.join(analyse_dir, i))]
        # If no analysis files, then return warning and don't make df
        if len(analysis_subdir_ls) == 0:
            outcome += "WARNING: no analysis fbf files made. Run `exp.analyse` first"
            return outcome
        # Reading in each fbf analysis df
        comb_df_ls = [
            AnalyseDf.read_feather(os.path.join(analyse_dir, analysis_subdir, "fbf", f"{name}.feather"))
            for analysis_subdir in analysis_subdir_ls
        ]
        # Making combined df from list of dfs
        comb_df = pd.concat(
            comb_df_ls,
            axis=1,
            keys=analysis_subdir_ls,
            names=[AnalyseCombinedDf.CN.ANALYSIS.value],
        )
        # Writing to file
        AnalyseCombinedDf.write_feather(comb_df, out_fp)
        # Returning outcome
        return outcome
