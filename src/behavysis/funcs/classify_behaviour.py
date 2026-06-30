"""Classify Behaviours."""

from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from behavysis.behaviour_classifier import BehaviourClassifier
from behavysis.constants import BEHAVIOUR, FRAME, PRED, PROB
from behavysis.models import ExperimentConfig
from behavysis.schemas import BEHAVIOUR_PREDICTED_SCHEMA, merge_bouts, write_df
from behavysis.utils.io_utils import file_exists_msg


def classify_behaviour(
    features_fp: Path,
    behaviour_fp: Path,
    config_fp: Path,
    *,
    overwrite: bool,
) -> None:
    """Given model config files and features df, classifies behaviour with ML model."""
    if not overwrite and behaviour_fp.exists():
        logger.warning(file_exists_msg(behaviour_fp))
        return
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    fps = config.auto.formatted_vid.fps
    model_config_ls = config.user.classify_behaviour

    # Read features (wide, pandas — BehaviourClassifier API expects pandas)

    features_df = pl.read_parquet(features_fp).to_pandas()

    behaviour_df_ls = []
    for model_config in model_config_ls:
        proj_dir = config.get_ref(model_config.proj_dir)
        behav_name = config.get_ref(model_config.behav_name)
        behav_model = BehaviourClassifier.load(proj_dir, behav_name)
        pcutoff = _get_pcutoff(
            config.get_ref(model_config.pcutoff),
            behav_model.config.pcutoff,
        )
        min_window_secs = config.get_ref(model_config.min_empty_window_secs)
        min_window_frames = int(np.round(min_window_secs * fps))

        # Run classifier pipeline (returns pandas DataFrame)
        behav_df_i = behav_model.pipeline_inference(features_df)

        # Convert to Polars long form
        df_pl = pl.DataFrame(
            {
                FRAME: pl.Series(behav_df_i.index.to_numpy(), dtype=pl.Int64),
                BEHAVIOUR: pl.lit(behav_name, dtype=pl.Utf8),
                PROB: pl.Series(behav_df_i.iloc[:, 0].to_numpy(), dtype=pl.Float64),
                PRED: pl.Series(
                    (behav_df_i.iloc[:, 0].to_numpy() > pcutoff).astype(int),
                    dtype=pl.Int64,
                ),
            },
        )

        # Merge close bouts
        df_pl = df_pl.with_columns(
            merge_bouts(df_pl.select(PRED).to_series(), min_window_frames).alias(PRED),
        )

        behaviour_df_ls.append(df_pl)
        logger.info("Completed %s classification.", behav_name)

    if len(behaviour_df_ls) == 0:
        return

    # Concatenate all behaviours vertically
    behaviour_df = pl.concat(behaviour_df_ls)
    write_df(behaviour_df, behaviour_fp, BEHAVIOUR_PREDICTED_SCHEMA)


def _get_pcutoff(pcutoff: float, model_pcutoff: float) -> float:
    """Check if the pcutoff is valid.

    Also check if the pcutoff is the special value `-1`, in which case
    `model_pcutoff` is used.
    """
    # Checking if pcutoff is -1, then using model_pcutoff
    if pcutoff == -1:
        # Checking if model_pcutoff is valid
        assert 0 <= model_pcutoff <= 1, (
            "pcutoff is relying on the model's pcutoff.\n"
            f"But the model's pcutoff is invalid: {model_pcutoff}.\n"
            "Must be between 0 and 1."
        )
        return model_pcutoff
    assert 0 <= pcutoff <= 1, (
        "pcutoff in config must be between 0 and 1, or the special value -1.\n"
        f"Instead it has value: {pcutoff}"
    )
    return pcutoff
