import os

from behavysis_pipeline.behav_classifier.behav_classifier import BehavClassifier
from behavysis_pipeline.df_classes.behav_df import (
    BehavPredictedDf,
    BehavScoredDf,
)
from behavysis_pipeline.df_classes.df_mixin import DFMixin
from behavysis_pipeline.pydantic_models.bouts import BoutStruct
from behavysis_pipeline.pydantic_models.configs import ExperimentConfigs
from behavysis_pipeline.utils.diagnostics_utils import file_exists_msg
from behavysis_pipeline.utils.logging_utils import get_io_obj_content, init_logger_with_io_obj
from behavysis_pipeline.utils.misc_utils import get_current_func_name


class Export:
    """__summary__"""

    @classmethod
    def df2df(
        cls,
        src_fp: str,
        dst_fp: str,
        overwrite: bool,
    ) -> str:
        """__summary__"""
        logger, io_obj = init_logger_with_io_obj(get_current_func_name())
        if not overwrite and os.path.exists(dst_fp):
            logger.warning(file_exists_msg(dst_fp))
            return get_io_obj_content(io_obj)
        df = DFMixin.read(src_fp)
        DFMixin.write(df, dst_fp)
        logger.info("df to df")
        return get_io_obj_content(io_obj)

    @classmethod
    def df2csv(
        cls,
        src_fp: str,
        dst_fp: str,
        overwrite: bool,
    ) -> str:
        """__summary__"""
        logger, io_obj = init_logger_with_io_obj(get_current_func_name())
        if not overwrite and os.path.exists(dst_fp):
            logger.warning(file_exists_msg(dst_fp))
            return get_io_obj_content(io_obj)
        df = DFMixin.read(src_fp)
        DFMixin.write_csv(df, dst_fp)
        logger.info("exported df to csv")
        return get_io_obj_content(io_obj)

    @classmethod
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
        logger, io_obj = init_logger_with_io_obj(get_current_func_name())
        if not overwrite and os.path.exists(dst_fp):
            logger.warning(file_exists_msg(dst_fp))
            return get_io_obj_content(io_obj)
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
        behavs_predicted_df = BehavPredictedDf.read(src_fp)
        behavs_scored_df = BehavScoredDf.predicted2scored(behavs_predicted_df, bouts_struct)
        BehavScoredDf.write(behavs_scored_df, dst_fp)
        logger.info("predicted_behavs to scored_behavs.")
        return get_io_obj_content(io_obj)

    @classmethod
    def boris2behav(
        cls,
        src_fp: str,
        dst_fp: str,
        configs_fp: str,
        behavs_ls: list[str],
        overwrite: bool,
    ) -> str:
        logger, io_obj = init_logger_with_io_obj(get_current_func_name())
        if not overwrite and os.path.exists(dst_fp):
            logger.warning(file_exists_msg(dst_fp))
            return get_io_obj_content(io_obj)
        # Reading the configs file
        configs = ExperimentConfigs.read_json(configs_fp)
        start_frame = configs.get_ref(configs.auto.start_frame)
        stop_frame = configs.get_ref(configs.auto.stop_frame) + 1
        # Importing the boris file to the Behav df format
        df = BehavScoredDf.import_boris_tsv(src_fp, behavs_ls, start_frame, stop_frame)
        BehavScoredDf.write(df, dst_fp)
        logger.info("boris tsv to behav")
        return get_io_obj_content(io_obj)
