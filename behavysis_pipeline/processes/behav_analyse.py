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

import numpy as np

from behavysis_core.constants import AnalysisCN, BehavCN, BehavColumns
from behavysis_core.data_models.experiment_configs import ExperimentConfigs
from behavysis_core.mixins.behav_mixin import BehavMixin
from behavysis_core.mixins.df_io_mixin import DFIOMixin
from behavysis_core.mixins.io_mixin import IOMixin

from .analyse_mixin import AggAnalyse, AnalyseMixin


class BehavAnalyse:
    """__summary__"""

    @staticmethod
    def behav_analysis(
        behavs_fp: str,
        analysis_dir: str,
        configs_fp: str,
        # bins: list,
        # summary_func: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> str:
        """
        Takes a behavs dataframe and generates a summary and binned version of the data.
        """
        outcome = ""
        name = IOMixin.get_name(behavs_fp)
        out_dir = os.path.join(analysis_dir, BehavAnalyse.behav_analysis.__name__)
        # Calculating the deltas (changes in body position) between each frame for the subject
        configs = ExperimentConfigs.read_json(configs_fp)
        fps, _, _, _, bins_ls, cbins_ls = AnalyseMixin.get_configs(configs)
        # Loading in dataframe
        behavs_df = BehavMixin.read_feather(behavs_fp)
        # Setting all na and -1 values to 0
        behavs_df = behavs_df.fillna(0).map(lambda x: np.maximum(0, x))
        # Getting the behaviour names and each user_behav for the behaviour
        # Not incl. the `pred` or `prob` (`prob` shouldn't be here anyway) column
        columns = np.isin(
            behavs_df.columns.get_level_values(BehavCN.OUTCOMES.value),
            [BehavColumns.PROB.value, BehavColumns.PRED.value],
            invert=True,
        )
        behavs_df = behavs_df.loc[:, columns]
        # Writing the behavs_df to the fbf file
        fbf_fp = os.path.join(out_dir, "fbf", f"{name}.feather")
        DFIOMixin.write_feather(behavs_df, fbf_fp)
        # Updating the column level names of behavs_df
        # (summary_binned_behavs only works this way)
        behavs_df.columns.names = DFIOMixin.enum_to_list(AnalysisCN)
        # Making the summary and binned dataframes
        AggAnalyse.summary_binned_behavs(
            behavs_df,
            out_dir,
            name,
            fps,
            bins_ls,
            cbins_ls,
        )
        return outcome
