import logging
from pathlib import Path

import numpy as np

from behavysis.df_classes.analysis_agg_df import AnalysisBinnedDf
from behavysis.df_classes.analysis_df import FBF, AnalysisDf
from behavysis.df_classes.behav_df import BehavScoredDf, BehavValues
from behavysis.models.experiment_configs import ExperimentConfigs
from behavysis.utils.io_utils import get_name

logger = logging.getLogger(__name__)


class AnalyseBehavs:
    @staticmethod
    def analyse_behavs(
        behavs_fp: Path,
        dst_dir: Path,
        configs_fp: Path,
    ) -> None:
        """Takes a behavs dataframe and generates a summary and binned version of the data."""
        name = get_name(behavs_fp)
        dst_subdir = dst_dir / "analyse_behavs"
        # Calculating the deltas (changes in body position) between each frame for the subject
        configs = ExperimentConfigs.model_validate_json(configs_fp.read_text())
        fps, _, _, _, bins_ls, cbins_ls = configs.get_analysis_configs()
        # Loading in dataframe
        behavs_df = BehavScoredDf.read(behavs_fp)
        # Setting all na and undetermined behav to non-behav
        behavs_df = behavs_df.fillna(0).replace(
            BehavValues.UNDETERMINED.value, BehavValues.NON_BEHAV.value
        )
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
        fbf_fp = dst_subdir / FBF / f"{name}.{AnalysisDf.IO}"
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
