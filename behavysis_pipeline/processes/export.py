import os

import numpy as np

from behavysis_pipeline.behav_classifier.behav_classifier import BehavClassifier
from behavysis_pipeline.df_classes.behav_df import BehavColumns, BehavDf
from behavysis_pipeline.df_classes.df_mixin import DFMixin
from behavysis_pipeline.df_classes.diagnostics_df import DiagnosticsMixin
from behavysis_pipeline.pydantic_models.experiment_configs import ExperimentConfigs
from behavysis_pipeline.utils.io_utils import IOMixin
from behavysis_pipeline.utils.logging_utils import func_decorator, init_logger


class Export:
    """__summary__"""

    logger = init_logger(__name__)

    @classmethod
    @func_decorator(logger)
    def feather2feather(
        cls,
        src_fp: str,
        out_fp: str,
        overwrite: bool,
    ) -> str:
        """__summary__"""
        if not overwrite and IOMixin.check_files_exist(out_fp):
            return DiagnosticsMixin.file_exists_msg(out_fp)
        # Reading file
        df = DFMixin.read_feather(src_fp)
        # Writing file
        DFMixin.write_feather(df, out_fp)
        # Returning outcome
        return "feather to feather\n"

    @classmethod
    @func_decorator(logger)
    def feather2csv(
        cls,
        src_fp: str,
        out_fp: str,
        overwrite: bool,
    ) -> str:
        """__summary__"""
        if not overwrite and IOMixin.check_files_exist(out_fp):
            return DiagnosticsMixin.file_exists_msg(out_fp)
        # Reading file
        df = DFMixin.read_feather(src_fp)
        # Writing file
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        df.to_csv(out_fp)
        # Returning outcome
        return "feather to csv\n"

    @classmethod
    @func_decorator(logger)
    def predbehavs2scoredbehavs(
        cls,
        src_fp: str,
        out_fp: str,
        configs_fp: str,
        overwrite: bool,
    ) -> str:
        """ """
        if not overwrite and IOMixin.check_files_exist(out_fp):
            return DiagnosticsMixin.file_exists_msg(out_fp)
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
            except (FileNotFoundError, OSError):
                outcome += f"WARNING: Model file {model.model_fp} not found. Skipping model.\n"
                continue
            behav_name_i = behav_model_i.configs.behaviour_name
            user_behavs_i = configs.get_ref(model.user_behavs)
            # Adding pred column
            out_df[(behav_name_i, BehavColumns.PRED.value)] = in_df[(behav_name_i, BehavColumns.PRED.value)].values
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
        BehavDf.write_feather(out_df, out_fp)
        # Returning outcome
        outcome += "predicted_behavs to scored_behavs.\n"
        return outcome

    @classmethod
    @func_decorator(logger)
    def boris2behav(
        cls,
        src_fp: str,
        out_fp: str,
        configs_fp: str,
        behavs_ls: list[str],
        overwrite: bool,
    ) -> str:
        if not overwrite and IOMixin.check_files_exist(out_fp):
            return DiagnosticsMixin.file_exists_msg(out_fp)
        # Reading the configs file
        configs = ExperimentConfigs.read_json(configs_fp)
        start_frame = configs.get_ref(configs.auto.start_frame)
        stop_frame = configs.get_ref(configs.auto.stop_frame) + 1
        # Importing the boris file to the Behav df format
        df = BehavDf.import_boris_tsv(src_fp, behavs_ls, start_frame, stop_frame)
        # Writing file
        BehavDf.write_feather(df, out_fp)
        # Returning outcome
        return "boris tsv to behav\n"
