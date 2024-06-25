import os

from behavysis_core.data_models.experiment_configs import ExperimentConfigs
from behavysis_core.mixins.behav_mixin import BehavMixin
from behavysis_core.mixins.df_io_mixin import DFIOMixin
from behavysis_core.mixins.io_mixin import IOMixin

from behavysis_pipeline.behav_classifier import BehavClassifier


class Export:
    """__summary__"""

    @staticmethod
    @IOMixin.overwrite_check()
    def feather_2_feather(src_fp: str, out_fp: str, overwrite: bool) -> str:
        """__summary__"""
        # Reading file
        df = DFIOMixin.read_feather(src_fp)
        # Writing file
        DFIOMixin.write_feather(df, out_fp)
        # Returning outcome
        return "feather to feather\n"

    @staticmethod
    @IOMixin.overwrite_check()
    def feather_2_csv(src_fp: str, out_fp: str, overwrite: bool) -> str:
        """__summary__"""
        # Reading file
        df = DFIOMixin.read_feather(src_fp)
        # Writing file
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        df.to_csv(out_fp)
        # Returning outcome
        return "feather to csv\n"

    @staticmethod
    @IOMixin.overwrite_check()
    def predbehav_2_scoredbehav(
        src_fp: str, out_fp: str, configs_fp: str, overwrite: bool
    ) -> str:
        """__summary__"""
        # Reading the configs file
        configs = ExperimentConfigs.read_json(configs_fp)
        models_ls = configs.user.classify_behaviours
        # Getting the behav_outcomes dict from the configs file
        behav_outcomes = {
            BehavClassifier.load(
                model.model_fp
            ).configs.behaviour_name: configs.get_ref(model.user_behavs)
            for model in models_ls
        }
        # Reading file
        in_df = BehavMixin.read_feather(src_fp)
        # Making the output df (with all user_behav outcome columns)
        out_df = BehavMixin.include_outcome_behavs(in_df, behav_outcomes)
        # Writing file
        DFIOMixin.write_feather(out_df, out_fp)
        # Returning outcome
        return "predicted_behavs to scored_behavs\n"

    @staticmethod
    @IOMixin.overwrite_check()
    def boris_2_behav(
        src_fp: str, out_fp: str, configs_fp: str, overwrite: bool
    ) -> str:
        # Reading the configs file
        configs = ExperimentConfigs.read_json(configs_fp)
        start_frame = configs.get_ref(configs.auto.start_frame)
        stop_frame = configs.get_ref(configs.auto.stop_frame) + 1
        # Importing the boris file to the Behav df format
        df = BehavMixin.import_boris_tsv(src_fp, start_frame, stop_frame)
        # Writing file
        DFIOMixin.write_feather(df, out_fp)
        # Returning outcome
        return "boris tsv to behav\n"
