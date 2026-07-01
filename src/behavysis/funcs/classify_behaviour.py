"""Classify Behaviours."""

import numpy as np
import polars as pl
from loguru import logger

from behavysis.behaviour_classifier import BehaviourClassifier
from behavysis.constants import BEHAVIOUR, FRAME, PRED, PROB
from behavysis.models import ExperimentConfig, ExperimentMetadata
from behavysis.schemas import BEHAVIOUR_PREDICTED_SCHEMA, merge_bouts


def classify_behaviour(
    features_df: pl.DataFrame,
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
) -> pl.DataFrame:
    """Given model config files and features df, classifies behaviour with ML model."""
    model_config_ls = config.require_classify_behaviour()

    behaviour_df_ls = []
    for model_config in model_config_ls:
        proj_dir = model_config.proj_dir
        behav_name = model_config.behav_name
        behav_model = BehaviourClassifier.load(proj_dir, behav_name)
        pcutoff = model_config.pcutoff or behav_model.config.pcutoff
        min_window_secs = model_config.min_empty_window_secs
        min_window_frames = int(np.round(min_window_secs * metadata.require_fps()))

        # Run classifier pipeline (returns pandas DataFrame)
        behav_df_i = behav_model.pipeline_inference(features_df.to_pandas())

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
        return pl.DataFrame(schema=BEHAVIOUR_PREDICTED_SCHEMA)

    # Concatenate all behaviours vertically
    return pl.concat(behaviour_df_ls)
