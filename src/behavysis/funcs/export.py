"""Export funcs."""

from pathlib import Path

import polars as pl
from loguru import logger

from behavysis.behaviour_classifier import BehaviourClassifier
from behavysis.models import BoutStruct, ExperimentConfig, ExperimentMetadata
from behavysis.schemas import (
    BEHAVIOUR_SCORED_BASE,
    import_boris_tsv,
    predicted2scored,
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
    behaviour_predicted_df: pl.DataFrame,
    config: ExperimentConfig,
) -> pl.DataFrame:
    """Converts a predicted_behaviour df to a scored_behaviour df.

    Namely:
    - Adds an "actual" column to the df.
        All predicted positive BEHAV frames are set to UNDETERMINED.
    - Adds user_defined columns to the df and sets all values to 0 (NON_BEHAV).
    """
    # Load configs
    models_ls = config.require_classify_behaviour()
    # Construct bouts_struct
    bouts_struct = []
    for model_config in models_ls:
        proj_dir = model_config.proj_dir
        behav_name = model_config.behav_name
        user_defined = model_config.user_defined
        BehaviourClassifier.load(proj_dir, behav_name)
        bouts_struct.append(BoutStruct(behav=behav_name, user_defined=user_defined))
    # Convert predicted df to scored df format
    return predicted2scored(behaviour_predicted_df, bouts_struct)


def boris2behaviour(
    src_fp: Path,
    dst_fp: Path,
    metadata_fp: Path,
    behaviour_ls: list[str],
    *,
    overwrite: bool,
) -> None:
    """Boris to Behaviour."""
    if not overwrite and dst_fp.exists():
        logger.warning(file_exists_msg(dst_fp))
        return

    metadata = ExperimentMetadata.model_validate_json(metadata_fp.read_text())

    df = import_boris_tsv(
        src_fp,
        behaviour_ls,
        metadata.require_start_frame(),
        metadata.require_stop_frame() + 1,
    )
    write_df(df, dst_fp, BEHAVIOUR_SCORED_BASE)
    logger.info("boris tsv to behav")
