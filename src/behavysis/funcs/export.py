from pathlib import Path

from loguru import logger

from behavysis.behav_classifier.behav_classifier import BehavClassifier
from behavysis.df_classes import DFMixin
from behavysis.df_classes.behav_df import (
    BehavPredictedDf,
    BehavScoredDf,
)
from behavysis.models.bouts import BoutStruct
from behavysis.models.experiment_configs import ExperimentConfigs
from behavysis.utils.io_utils import file_exists_msg


def df2df(
    src_fp: Path,
    dst_fp: Path,
    *,
    overwrite: bool,
) -> None:
    """Convert dataframe between formats based on file extensions."""
    if not overwrite and dst_fp.exists():
        logger.warning(file_exists_msg(dst_fp))
        return
    df = DFMixin.read(src_fp)
    DFMixin.write(df, dst_fp)
    logger.info("df to df")


def df2csv(
    src_fp: Path,
    dst_fp: Path,
    *,
    overwrite: bool,
) -> None:
    """Export dataframe to CSV format."""
    if not overwrite and dst_fp.exists():
        logger.warning(file_exists_msg(dst_fp))
        return
    df = DFMixin.read(src_fp)
    DFMixin.write(df, dst_fp, fmt="csv")
    logger.info("exported df to csv")


def predictedbehavs2scoredbehavs(
    src_fp: Path,
    dst_fp: Path,
    configs_fp: Path,
    *,
    overwrite: bool,
) -> None:
    """Converts a predicted_behavs df to a scored_behavs df.

    Namely:
    - Adds an "actual" column to the df.
        All predicted positive BEHAV frames are set to UNDETERMINED.
    - Adds user_defined columns to the df and sets all values to 0 (NON_BEHAV).
    """
    if not overwrite and dst_fp.exists():
        logger.warning(file_exists_msg(dst_fp))
        return
    # Reading the configs file
    configs = ExperimentConfigs.model_validate_json(configs_fp.read_text())
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
        bouts_struct.append(BoutStruct(behav=behav_name, user_defined=user_defined))
    # Getting scored behavs df from predicted behavs df and bouts_struct
    behavs_predicted_df = BehavPredictedDf.read(src_fp)
    behavs_scored_df = BehavScoredDf.predicted2scored(behavs_predicted_df, bouts_struct)
    BehavScoredDf.write(behavs_scored_df, dst_fp)
    logger.info("predicted_behavs to scored_behavs.")


def boris2behav(
    src_fp: Path,
    dst_fp: Path,
    configs_fp: Path,
    behavs_ls: list[str],
    *,
    overwrite: bool,
) -> None:
    """Boris to Behaviour."""
    if not overwrite and dst_fp.exists():
        logger.warning(file_exists_msg(dst_fp))
        return
    # Reading the configs file
    configs = ExperimentConfigs.model_validate_json(configs_fp.read_text())
    start_frame = configs.get_ref(configs.auto.start_frame)
    stop_frame = configs.get_ref(configs.auto.stop_frame) + 1
    # Importing the boris file to the Behav df format
    df = BehavScoredDf.import_boris_tsv(src_fp, behavs_ls, start_frame, stop_frame)
    BehavScoredDf.write(df, dst_fp)
    logger.info("boris tsv to behav")
