"""Export funcs."""

from pathlib import Path

import polars as pl
from loguru import logger

from behavysis.behaviour_classifier import BehaviourClassifier
from behavysis.models import BoutStruct, ExperimentConfig
from behavysis.schemas import (
    BEHAVIOUR_PREDICTED_SCHEMA,
    BEHAVIOUR_SCORED_BASE,
    import_boris_tsv,
    predicted2scored,
    read_df,
    write_df,
)
from behavysis.utils.io_utils import file_exists_msg


def df2df(
    src_fp: Path,
    dst_fp: Path,
    *,
    overwrite: bool,
) -> None:
    """Copy dataframe between locations/formats."""
    if not overwrite and dst_fp.exists():
        logger.warning(file_exists_msg(dst_fp))
        return
    df = pl.read_parquet(src_fp)
    dst_fp.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dst_fp)
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
    df = pl.read_parquet(src_fp)
    dst_fp.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(dst_fp)
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
    # Load configs
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    models_ls = config.user.classify_behaviour
    # Construct bouts_struct
    bouts_struct = []
    for model_config in models_ls:
        proj_dir = config.get_ref(model_config.proj_dir)
        behav_name = config.get_ref(model_config.behav_name)
        user_defined = config.get_ref(model_config.user_defined)
        BehaviourClassifier.load(proj_dir, behav_name)
        bouts_struct.append(BoutStruct(behav=behav_name, user_defined=user_defined))
    # Read predicted df
    behaviour_predicted_df = read_df(src_fp, BEHAVIOUR_PREDICTED_SCHEMA)
    # Convert predicted df to scored df format
    behaviour_scored_df = predicted2scored(behaviour_predicted_df, bouts_struct)
    # Build dynamic schema: base + user_defined columns
    scored_schema = dict(BEHAVIOUR_SCORED_BASE)
    for bs in bouts_struct:
        for col in bs.user_defined:
            scored_schema[col] = pl.Int64
    # Write scored df to file
    write_df(behaviour_scored_df, dst_fp, scored_schema)
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

    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    start_frame = config.get_ref(config.auto.start_frame)
    stop_frame = config.get_ref(config.auto.stop_frame) + 1

    df = import_boris_tsv(src_fp, behaviour_ls, start_frame, stop_frame)
    write_df(df, dst_fp, BEHAVIOUR_SCORED_BASE)
    logger.info("boris tsv to behav")
