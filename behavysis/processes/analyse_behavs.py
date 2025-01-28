import os

import numpy as np

from behavysis.df_classes.analysis_agg_df import AnalysisBinnedDf
from behavysis.df_classes.analysis_df import FBF, AnalysisDf
from behavysis.df_classes.behav_df import BehavScoredDf, BehavValues
from behavysis.pydantic_models.configs import ExperimentConfigs
from behavysis.utils.io_utils import get_name
from behavysis.utils.logging_utils import get_io_obj_content, init_logger_io_obj
from behavysis.utils.misc_utils import get_func_name_in_stack

###################################################################################################
#               ANALYSIS API FUNCS
###################################################################################################


class AnalyseBehavs:
    @staticmethod
    def analyse_behavs(
        behavs_fp: str,
        dst_dir: str,
        configs_fp: str,
    ) -> str:
        """
        Takes a behavs dataframe and generates a summary and binned version of the data.
        """
        logger, io_obj = init_logger_io_obj()
        f_name = get_func_name_in_stack()
        name = get_name(behavs_fp)
        dst_subdir = os.path.join(dst_dir, f_name)
        # Calculating the deltas (changes in body position) between each frame for the subject
        configs = ExperimentConfigs.read_json(configs_fp)
        fps, _, _, _, bins_ls, cbins_ls = configs.get_analysis_configs()
        # Loading in dataframe
        behavs_df = BehavScoredDf.read(behavs_fp)
        # Setting all na and undetermined behav to non-behav
        behavs_df = behavs_df.fillna(0).replace(BehavValues.UNDETERMINED.value, BehavValues.NON_BEHAV.value)
        # Getting the behaviour names and each user_defined for the behaviour
        # Not incl. the `pred` or `prob` (`prob` shouldn't be here anyway) columns
        columns = np.isin(
            behavs_df.columns.get_level_values(BehavScoredDf.CN.OUTCOMES.value),
            [BehavScoredDf.OutcomesCols.PRED.value],
            invert=True,
        )
        behavs_df = behavs_df.loc[:, columns]
        behavs_df = AnalysisDf.basic_clean(behavs_df)
        # Writing the behavs_df to the fbf file
        fbf_fp = os.path.join(dst_subdir, FBF, f"{name}.{AnalysisDf.IO}")
        AnalysisDf.write(behavs_df, fbf_fp)
        # Making the summary and binned dataframes
        AnalysisBinnedDf.summary_binned_behavs(
            behavs_df,
            dst_subdir,
            name,
            fps,
            bins_ls,
            cbins_ls,
        )
        return get_io_obj_content(io_obj)
