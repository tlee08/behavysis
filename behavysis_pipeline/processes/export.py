import os

import numpy as np
from behavysis_core.df_classes.behav_df import BehavColumns, BehavDf
from behavysis_core.df_classes.df_mixin import DFMixin
from behavysis_core.mixins.io_mixin import IOMixin
from behavysis_core.pydantic_models.experiment_configs import ExperimentConfigs

from behavysis_pipeline.behav_classifier.behav_classifier import BehavClassifier


class Export:
    """__summary__"""

    @staticmethod
    @IOMixin.overwrite_check()
    def feather2feather(
        src_fp: str,
        out_fp: str,
        overwrite: bool,
    ) -> str:
        """__summary__"""
        # Reading file
        df = DFMixin.read_feather(src_fp)
        # Writing file
        DFMixin.write_feather(df, out_fp)
        # Returning outcome
        return "feather to feather\n"

    @staticmethod
    @IOMixin.overwrite_check()
    def feather2csv(
        src_fp: str,
        out_fp: str,
        overwrite: bool,
    ) -> str:
        """__summary__"""
        # Reading file
        df = DFMixin.read_feather(src_fp)
        # Writing file
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        df.to_csv(out_fp)
        # Returning outcome
        return "feather to csv\n"

    @staticmethod
    @IOMixin.overwrite_check()
    def predbehavs2scoredbehavs(
        src_fp: str,
        out_fp: str,
        configs_fp: str,
        overwrite: bool,
    ) -> str:
        """ """
        outcome = ""
        # Reading the configs file
        configs = ExperimentConfigs.read_json(configs_fp)
        models_ls = configs.user.classify_behaviours
        # Reading file
        in_df = BehavDf.read_feather(src_fp)
        # Making out_df
        out_df = BehavDf.init_df(in_df.index)
        # Getting the behav_outcomes dict from the configs file
        for model in models_ls:
            # Loading behav_model
            try:
                behav_model_i = BehavClassifier.load(model.model_fp)
            except FileNotFoundError:
                outcome += "WARNING: Model file not found. Skipping model.\n"
                continue
            behav_name_i = behav_model_i.configs.behaviour_name
            user_behavs_i = configs.get_ref(model.user_behavs)
            # Adding pred column
            out_df[(behav_name_i, BehavColumns.PRED.value)] = in_df[
                (behav_name_i, BehavColumns.PRED.value)
            ].values
            # Adding actual column
            out_df[(behav_name_i, BehavColumns.ACTUAL.value)] = in_df[
                (behav_name_i, BehavColumns.PRED.value)
            ].values * np.array(-1)
            # Adding user_behav columns
            for user_behavs_i_j in user_behavs_i:
                out_df[(behav_name_i, user_behavs_i_j)] = 0
        # Ordering by "behaviours" level
        out_df = out_df.sort_index(axis=1, level=BehavDf.CN.BEHAVIOURS.value)
        # Writing file
        DFMixin.write_feather(out_df, out_fp)
        # Returning outcome
        outcome += "predicted_behavs to scored_behavs.\n"
        return outcome

    @staticmethod
    @IOMixin.overwrite_check()
    def boris2behav(
        src_fp: str,
        out_fp: str,
        configs_fp: str,
        behavs_ls: list[str],
        overwrite: bool,
    ) -> str:
        # Reading the configs file
        configs = ExperimentConfigs.read_json(configs_fp)
        start_frame = configs.get_ref(configs.auto.start_frame)
        stop_frame = configs.get_ref(configs.auto.stop_frame) + 1
        # Importing the boris file to the Behav df format
        df = BehavDf.import_boris_tsv(src_fp, behavs_ls, start_frame, stop_frame)
        # Writing file
        DFMixin.write_feather(df, out_fp)
        # Returning outcome
        return "boris tsv to behav\n"
