import os

from behavysis_core.constants import BehavColumns
from behavysis_core.data_models.experiment_configs import ExperimentConfigs
from behavysis_core.mixins.behav_mixin import BehavMixin
from behavysis_core.mixins.df_io_mixin import DFIOMixin
from behavysis_core.mixins.io_mixin import IOMixin

from behavysis_pipeline.behav_classifier import BehavClassifier


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
        # Reading file
        in_df = BehavMixin.read_feather(in_fp)
        # Keeping the `actual`, `pred`, and all user_behavs columns
        out_df = BehavMixin.init_df(in_df.index)
        a = BehavColumns.ACTUAL.value
        p = BehavColumns.PRED.value
        for model_config in models_ls:
            # Getting configs
            model_fp = configs.get_ref(model_config.model_fp)
            model = BehavClassifier.load(model_fp)
            behav = configs.get_ref(model.configs.behaviour_name)
            user_behavs = configs.get_ref(model_config.user_behavs)
            # Adding pred and actual columns
            out_df[(behav, p)] = in_df[(behav, p)].values
            out_df[(behav, a)] = 0
            # Adding user behavs to the "outcome" column level
            for i in user_behavs:
                out_df[(behav, i)] = 0
        # Ordering by "behaviours" level
        out_df = out_df.sort_index(axis=1, level="behaviours")
        # Writing file
        DFIOMixin.write_feather(out_df, out_fp)
        # Returning outcome
        return "predicted_behavs to scored_behavs\n"
