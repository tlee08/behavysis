from pathlib import Path

from loguru import logger

from behavysis.behav_classifier.behav_classifier import BehaviourClassifier
from behavysis.df_classes import BehaviourPredictedDf, BehaviourScoredDf, DFMixin
from behavysis.models import BoutStruct, ExperimentConfig
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


def predictedbehaviour2scoredbehaviour(
    src_fp: Path,
    dst_fp: Path,
    config_fp: Path,
    *,
    overwrite: bool,
) -> None:
    """Converts a predicted_behaviour df to a scored_behaviour df.

    Namely:
    - Adds an "actual" column to the df.
        All predicted positive BEHAV frames are set to UNDETERMINED.
    - Adds user_defined columns to the df and sets all values to 0 (NON_BEHAV).
    """
    if not overwrite and dst_fp.exists():
        logger.warning(file_exists_msg(dst_fp))
        return
    # Reading the config file
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    models_ls = config.user.classify_behaviour
    # Getting the behav_outcomes dict from the config file
    bouts_struct = []
    for model_config in models_ls:
        proj_dir = config.get_ref(model_config.proj_dir)
        behav_name = config.get_ref(model_config.behav_name)
        user_defined = config.get_ref(model_config.user_defined)
        # Ensuring model exists
        BehaviourClassifier.load(proj_dir, behav_name)
        # Adding to bouts_struct
        bouts_struct.append(BoutStruct(behav=behav_name, user_defined=user_defined))
    # Getting scored behaviour df from predicted behaviour df and bouts_struct
    behaviour_predicted_df = BehaviourPredictedDf.read(src_fp)
    behaviour_scored_df = BehaviourScoredDf.predicted2scored(
        behaviour_predicted_df, bouts_struct
    )
    BehaviourScoredDf.write(behaviour_scored_df, dst_fp)
    logger.info("predicted_behaviour to scored_behaviour.")


def boris2behaviour(
    src_fp: Path,
    dst_fp: Path,
    config_fp: Path,
    behaviour_ls: list[str],
    *,
    overwrite: bool,
) -> None:
    """Boris to Behaviour."""
    if not overwrite and dst_fp.exists():
        logger.warning(file_exists_msg(dst_fp))
        return
    # Reading the config file
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    start_frame = config.get_ref(config.auto.start_frame)
    stop_frame = config.get_ref(config.auto.stop_frame) + 1
    # Importing the boris file to the Behav df format
    df = BehaviourScoredDf.import_boris_tsv(
        src_fp, behaviour_ls, start_frame, stop_frame
    )
    BehaviourScoredDf.write(df, dst_fp)
    logger.info("boris tsv to behav")
