import logging
from pathlib import Path

import pandas as pd

from behavysis.df_classes.analysis_combined_df import AnalysisCombinedDf
from behavysis.df_classes.analysis_df import FBF, AnalysisDf
from behavysis.utils.diagnostics_utils import file_exists_msg
from behavysis.utils.io_utils import get_name

logger = logging.getLogger(__name__)


class CombineAnalysis:
    @classmethod
    def combine_analysis(
        cls,
        analysis_dir: Path,
        analysis_combined_fp: Path,
        configs_fp: Path,
        overwrite: bool,
    ) -> None:
        """Concatenates across columns the frame-by-frame dataframes for all analysis subdirectories
        and saves this in a single dataframe.
        """
        if not overwrite and analysis_combined_fp:
            logger.warning(file_exists_msg(analysis_combined_fp))
            return
        name = get_name(configs_fp)
        # For each analysis subdir, combining fbf files
        analysis_subdir_ls = [
            i for i in analysis_dir.iterdir() if (analysis_dir / i).is_dir()
        ]
        # If no analysis files, then return warning and don't make df
        if len(analysis_subdir_ls) == 0:
            logger.warning("no analysis fbf files made. Run `exp.analyse` first")
            return
        # Reading in each fbf analysis df
        comb_df_ls = [
            AnalysisDf.read(
                analysis_dir / analysis_subdir / FBF / f"{name}.{AnalysisDf.IO}"
            )
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
