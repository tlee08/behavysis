import functools
import os
from typing import Callable

import pandas as pd
from behavysis_core.constants import BEHAV_COLUMN_NAMES, BehavColumns
from behavysis_core.data_models.experiment_configs import ExperimentConfigs
from behavysis_core.mixins.behaviour_mixin import BehaviourMixin
from behavysis_core.mixins.df_io_mixin import DFIOMixin
from behavysis_core.mixins.io_mixin import IOMixin


class Export:
    """__summary__"""

    @staticmethod
    @IOMixin.overwrite_check()
    def feather_2_feather(in_fp: str, out_fp: str, overwrite: bool) -> str:
        """__summary__"""
        # Reading file
        df = DFIOMixin.read_feather(in_fp)
        # Writing file
        DFIOMixin.write_feather(df, out_fp)
        # Returning outcome
        return "feather to feather\n"

    @staticmethod
    @IOMixin.overwrite_check()
    def feather_2_csv(in_fp: str, out_fp: str, overwrite: bool) -> str:
        """__summary__"""
        # Reading file
        df = DFIOMixin.read_feather(in_fp)
        # Writing file
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        df.to_csv(out_fp)
        # Returning outcome
        return "feather to csv\n"

    @staticmethod
    @IOMixin.overwrite_check()
    def behaviour_export(
        in_fp: str, out_fp: str, configs_fp: str, overwrite: bool
    ) -> str:
        """__summary__"""
        # Reading the configs file
        configs = ExperimentConfigs.read_json(configs_fp)
        user_behavs = configs.user.classify_behaviours.user_behavs
        # Reading file
        df = DFIOMixin.read_feather(in_fp)
        # Adding in behaviour columns
        df = BehaviourMixin.frames_add_behaviour(df, user_behavs)
        # Keeping the `actual`, `pred`, and all user_behavs columns
        a = BehavColumns.ACTUAL.value
        p = BehavColumns.PRED.value
        for behav in behavs:
            analysis_df[(behav, a)] = behavs_df[(behav, a)].values
            analysis_df[(behav, p)] = behavs_df[(behav, p)].values
            for i in user_behavs:
                analysis_df[(behav, i)] = behavs_df[(behav, i)].values
        # Ordering by "behaviours" level
        df = df.sort_index(axis=1, level="behaviours")
        # Writing file
        DFIOMixin.write_feather(df, out_fp)
        # Returning outcome
        return "prediced_behavs to scored_behavs\n"
