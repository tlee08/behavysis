import os

import pandas as pd

from behavysis_pipeline.df_classes.analysis_combined_df import AnalysisCombinedDf
from behavysis_pipeline.df_classes.analysis_df import FBF, AnalysisDf
from behavysis_pipeline.utils.diagnostics_utils import file_exists_msg
from behavysis_pipeline.utils.io_utils import get_name
from behavysis_pipeline.utils.logging_utils import get_io_obj_content, init_logger_io_obj

###################################################################################################
#               ANALYSIS API FUNCS
###################################################################################################


class CombineAnalysis:
    @classmethod
    def combine_analysis(
        cls,
        analysis_dir: str,
        analysis_combined_fp: str,
        configs_fp: str,
        overwrite: bool,
    ) -> str:
        """
        Concatenates across columns the frame-by-frame dataframes for all analysis subdirectories
        and saves this in a single dataframe.
        """
        logger, io_obj = init_logger_io_obj()
        if not overwrite and os.path.exists(analysis_combined_fp):
            logger.warning(file_exists_msg(analysis_combined_fp))
            return get_io_obj_content(io_obj)
        name = get_name(configs_fp)
        # For each analysis subdir, combining fbf files
        analysis_subdir_ls = [i for i in os.listdir(analysis_dir) if os.path.isdir(os.path.join(analysis_dir, i))]
        # If no analysis files, then return warning and don't make df
        if len(analysis_subdir_ls) == 0:
            logger.warning("no analysis fbf files made. Run `exp.analyse` first")
            return get_io_obj_content(io_obj)
        # Reading in each fbf analysis df
        comb_df_ls = [
            AnalysisDf.read(os.path.join(analysis_dir, analysis_subdir, FBF, f"{name}.{AnalysisDf.IO}"))
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
        AnalysisCombinedDf.write(comb_df, analysis_combined_fp)
        return get_io_obj_content(io_obj)
