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
from behavysis_core.constants import BehavColumns
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
        # Calculating more parameters
        user_behavs = configs.get_ref(configs.user.classify_behaviours.user_behavs)
        # Loading in dataframe
        behavs_df = BehavMixin.read_feather(behavs_fp)
        # Setting all na and -1 values to 0
        behavs_df = behavs_df.fillna(0).map(lambda x: np.maximum(0, x))
        # Getting the behaviour names
        behavs = behavs_df.columns.unique("behaviours")
        # Converting to the analysis dataframe format (with specific index and column levels)
        analysis_df = AnalyseMixin.init_df(behavs_df.index)
        # Keeping the `actual` and all user_behavs columns
        a = BehavColumns.ACTUAL.value
        for behav in behavs:
            analysis_df[(behav, a)] = behavs_df[(behav, a)].values
            for i in user_behavs:
                analysis_df[(behav, i)] = behavs_df[(behav, i)].values
        # Saving analysis_df
        fbf_fp = os.path.join(out_dir, "fbf", f"{name}.feather")
        DFIOMixin.write_feather(analysis_df, fbf_fp)

        # Making the summary and binned dataframes
        AggAnalyse.summary_binned_behavs(
            analysis_df,
            out_dir,
            name,
            fps,
            bins_ls,
            cbins_ls,
        )
        return outcome
