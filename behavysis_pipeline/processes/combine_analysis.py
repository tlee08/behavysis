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

import os

import pandas as pd

from behavysis_pipeline.df_classes.analyse_combined_df import AnalyseCombinedDf
from behavysis_pipeline.df_classes.analyse_df import AnalyseDf
from behavysis_pipeline.utils.diagnostics_utils import file_exists_msg
from behavysis_pipeline.utils.io_utils import get_name
from behavysis_pipeline.utils.logging_utils import init_logger_with_io_obj, io_obj_to_msg
from behavysis_pipeline.utils.misc_utils import get_current_funct_name

###################################################################################################
#               ANALYSIS API FUNCS
###################################################################################################


class CombineAnalysis:
    """__summary__"""

    @classmethod
    def combine_analysis(
        cls,
        analyse_dir: str,
        out_fp: str,
        configs_fp: str,
        overwrite: bool,
    ) -> str:
        """
        Concatenates across columns the frame-by-frame dataframes for all analysis subdirectories
        and saves this in a single dataframe.
        """
        logger, io_obj = init_logger_with_io_obj(get_current_funct_name())
        if not overwrite and os.path.exists(out_fp):
            logger.warning(file_exists_msg(out_fp))
            return io_obj_to_msg(io_obj)
        name = get_name(configs_fp)
        # For each analysis subdir, combining fbf files
        analysis_subdir_ls = [i for i in os.listdir(analyse_dir) if os.path.isdir(os.path.join(analyse_dir, i))]
        # If no analysis files, then return warning and don't make df
        if len(analysis_subdir_ls) == 0:
            logger.warning("no analysis fbf files made. Run `exp.analyse` first")
            return io_obj_to_msg(io_obj)
        # Reading in each fbf analysis df
        comb_df_ls = [
            AnalyseDf.read(os.path.join(analyse_dir, analysis_subdir, "fbf", f"{name}.{AnalyseDf.IO}"))
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
        AnalyseCombinedDf.write(comb_df, out_fp)
        return io_obj_to_msg(io_obj)
