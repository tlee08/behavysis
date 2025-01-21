import os

from behavysis_pipeline.behav_classifier.behav_classifier import BehavClassifier
from behavysis_pipeline.df_classes.behav_df import (
    BehavPredictedDf,
    BehavScoredDf,
)
from behavysis_pipeline.df_classes.df_mixin import DFMixin
from behavysis_pipeline.pydantic_models.bouts import BoutStruct
from behavysis_pipeline.pydantic_models.experiment_configs import ExperimentConfigs
from behavysis_pipeline.utils.diagnostics_utils import file_exists_msg
from behavysis_pipeline.utils.logging_utils import init_logger, logger_func_decorator


class Export:
    """__summary__"""

    logger = init_logger(__name__)

    @classmethod
    def df2df(
        cls,
        src_fp: str,
        dst_fp: str,
        overwrite: bool,
    ) -> str:
        """__summary__"""
        if not overwrite and os.path.exists(dst_fp):
            return file_exists_msg(dst_fp)
        # Reading file
        df = DFMixin.read(src_fp)
        # Writing file
        DFMixin.write(df, dst_fp)
        # Returning outcome
        return "df to df\n"

    @classmethod
    @logger_func_decorator(logger)
    def df2csv(
        cls,
        src_fp: str,
        dst_fp: str,
        overwrite: bool,
    ) -> str:
        """__summary__"""
        if not overwrite and os.path.exists(dst_fp):
            return file_exists_msg(dst_fp)
        # Reading file
        df = DFMixin.read(src_fp)
        # Writing file
        DFMixin.write_csv(df, dst_fp)
        # Returning outcome
        return "exported df to csv\n"

    @classmethod
    @logger_func_decorator(logger)
    def predictedbehavs2scoredbehavs(
        cls,
        src_fp: str,
        dst_fp: str,
        configs_fp: str,
        overwrite: bool,
    ) -> str:
        """
        Converts a predicted_behavs df to a scored_behavs df.
        Namely:
        - Adds an "actual" column to the df. All predicted positive BEHAV frames are set to UNDETERMINED.
        - Adds user_defined columns to the df and sets all values to 0 (NON_BEHAV).
        """
        if not overwrite and os.path.exists(dst_fp):
            return file_exists_msg(dst_fp)
        outcome = ""
        # Reading the configs file
        configs = ExperimentConfigs.read_json(configs_fp)
        models_ls = configs.user.classify_behavs
        # Getting the behav_outcomes dict from the configs file
        bouts_struct = []
        for model_config in models_ls:
            proj_dir = configs.get_ref(model_config.proj_dir)
            behav_name = configs.get_ref(model_config.behav_name)
            user_defined = configs.get_ref(model_config.user_defined)
            # Ensuring model exists
            BehavClassifier.load(proj_dir, behav_name)
            # Adding to bouts_struct
            bouts_struct.append(
                BoutStruct.model_validate(
                    {
                        BehavScoredDf.BoutCols.BEHAV.value: behav_name,
                        BehavScoredDf.BoutCols.USER_DEFINED.value: user_defined,
                    }
                )
            )
        # Getting scored behavs df from predicted behavs df and bouts_struct
        src_df = BehavPredictedDf.read(src_fp)
        dst_df = BehavScoredDf.predicted2scored(src_df, bouts_struct)
        # Writing file
        BehavScoredDf.write(dst_df, dst_fp)
        # Returning outcome
        outcome += "predicted_behavs to scored_behavs.\n"
        return outcome

    @classmethod
    @logger_func_decorator(logger)
    def boris2behav(
        cls,
        src_fp: str,
        dst_fp: str,
        configs_fp: str,
        behavs_ls: list[str],
        overwrite: bool,
    ) -> str:
        if not overwrite and os.path.exists(dst_fp):
            return file_exists_msg(dst_fp)
        # Reading the configs file
        configs = ExperimentConfigs.read_json(configs_fp)
        start_frame = configs.get_ref(configs.auto.start_frame)
        stop_frame = configs.get_ref(configs.auto.stop_frame) + 1
        # Importing the boris file to the Behav df format
        df = BehavScoredDf.import_boris_tsv(src_fp, behavs_ls, start_frame, stop_frame)
        # Writing file
        BehavScoredDf.write(df, dst_fp)
        # Returning outcome
        return "boris tsv to behav\n"
