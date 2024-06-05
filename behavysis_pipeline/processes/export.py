import os

from behavysis_core.constants import BEHAV_CN, BehavColumns
from behavysis_core.data_models.experiment_configs import ExperimentConfigs
from behavysis_core.mixins.behav_mixin import BehavMixin
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
    def behav_export(in_fp: str, out_fp: str, configs_fp: str, overwrite: bool) -> str:
        """__summary__"""
        # Reading the configs file
        configs = ExperimentConfigs.read_json(configs_fp)
        user_behavs = configs.get_ref(configs.user.classify_behaviours.user_behavs)
        # Reading file
        in_df = BehavMixin.read_feather(in_fp)
        # Adding in behaviour columns
        in_df = BehavMixin.frames_add_behaviour(in_df, user_behavs)
        # Keeping the `actual`, `pred`, and all user_behavs columns
        out_df = BehavMixin.init_df(in_df.index)
        a = BehavColumns.ACTUAL.value
        p = BehavColumns.PRED.value
        for behav in in_df.columns.unique(BEHAV_CN[0]):
            out_df[(behav, a)] = in_df[(behav, a)].values
            out_df[(behav, p)] = in_df[(behav, p)].values
            for i in user_behavs:
                out_df[(behav, i)] = in_df[(behav, i)].values
        # Ordering by "behaviours" level
        out_df = out_df.sort_index(axis=1, level="behaviours")
        # Writing file
        DFIOMixin.write_feather(out_df, out_fp)
        # Returning outcome
        return "prediced_behavs to scored_behavs\n"
