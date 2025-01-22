from __future__ import annotations

import os

import pandas as pd

from behavysis_pipeline.df_classes.analysis_combined_df import AnalysisCombinedDf
from behavysis_pipeline.df_classes.analysis_df import AnalysisDf
from behavysis_pipeline.utils.diagnostics_utils import file_exists_msg
from behavysis_pipeline.utils.io_utils import get_name
from behavysis_pipeline.utils.logging_utils import init_logger

###################################################################################################
#               ANALYSIS API FUNCS
###################################################################################################


class CombineAnalysis:
    """__summary__"""

    logger = init_logger(__name__)

    @classmethod
    def combine_analysis(
        cls,
        analysis_dir: str,
        out_fp: str,
        configs_fp: str,
        # bins: list,
        # summary_func: Callable[[pd.DataFrame], pd.DataFrame],
        overwrite: bool,
    ) -> str:
        """
        Concatenates across columns the frame-by-frame dataframes for all analysis subdirectories
        and saves this in a single dataframe.
        """
        if not overwrite and os.path.exists(out_fp):
            return file_exists_msg(out_fp)
        outcome = ""
        name = get_name(configs_fp)
        # For each analysis subdir, combining fbf files
        analysis_subdir_ls = [i for i in os.listdir(analysis_dir) if os.path.isdir(os.path.join(analysis_dir, i))]
        # If no analysis files, then return warning and don't make df
        if len(analysis_subdir_ls) == 0:
            outcome += "WARNING: no analysis fbf files made. Run `exp.analyse` first"
            return outcome
        # Reading in each fbf analysis df
        comb_df_ls = [
            AnalysisDf.read(os.path.join(analysis_dir, analysis_subdir, "fbf", f"{name}.{AnalysisDf.IO}"))
            for analysis_subdir in analysis_subdir_ls
        ]
        # Making combined df from list of dfs
        comb_df = pd.concat(
            comb_df_ls,
            axis=1,
            keys=analysis_subdir_ls,
            names=[AnalysisCombinedDf.CN.ANALYSIS.value],
        )
        # Writing to file
        AnalysisCombinedDf.write(comb_df, out_fp)
        # Returning outcome
        return outcome
