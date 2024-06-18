import os

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
        models_ls = configs.user.classify_behaviours
        # Getting the behav_outcomes dict from the configs file
        behav_outcomes = {
            configs.get_ref(model.configs.behaviour_name): configs.get_ref(
                model.configs.user_behavs
            )
            for model in models_ls
        }
        # Reading file
        in_df = BehavMixin.read_feather(in_fp)
        # Making the output df (with all user_behav outcome columns)
        BehavMixin.include_outcome_behavs(in_df, behav_outcomes)
        # Writing file
        DFIOMixin.write_feather(in_df, out_fp)
        # Returning outcome
        return "predicted_behavs to scored_behavs\n"
