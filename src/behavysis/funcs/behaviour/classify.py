"""Classify Behaviours."""

import numpy as np
import polars as pl
from loguru import logger

from behavysis.behaviour_classifier import BehaviourClassifier
from behavysis.constants import BEHAVIOUR, FRAME, PRED, PROB
from behavysis.models import (
    ClassifyBehaviourConfig,
    ExperimentConfig,
    ExperimentMetadata,
    ExtractFeaturesConfig,
)
from behavysis.schemas import BEHAVIOUR_PREDICTED_SCHEMA
from behavysis.transforms.behaviour import merge_bouts


def classify_behaviour(
    features_df: pl.DataFrame,
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
) -> pl.DataFrame:
    """Classify behaviour using trained models.

    Validates that each model's bodypoint config matches the experiment's
    extract_features config before classifying.
    """
    model_config_ls = config.require_classify_behaviour()
    feat_cfg = config.require_extract_features()

    behaviour_df_ls = []
    for model_config in model_config_ls:
        _validate_bodypoint_match(feat_cfg, model_config)

        proj_dir = model_config.proj_dir
        behaviour_name = model_config.behaviour_name
        model_type = model_config.model_type

        behaviour_model = BehaviourClassifier.load(
            proj_dir, behaviour_name, model_type=model_type,
        )
        pcutoff = model_config.pcutoff or behaviour_model.config.pcutoff
        min_window_secs = model_config.min_empty_window_secs
        min_window_frames = int(np.round(min_window_secs * metadata.require_fps()))

        behaviour_df_i = behaviour_model.predict(features_df.to_pandas())

        df_pl = pl.DataFrame(
            {
                FRAME: pl.Series(behaviour_df_i.index.to_numpy(), dtype=pl.Int64),
                BEHAVIOUR: pl.lit(behaviour_name, dtype=pl.Utf8),
                PROB: pl.Series(behaviour_df_i.iloc[:, 0].to_numpy(), dtype=pl.Float64),
                PRED: pl.Series(
                    (behaviour_df_i.iloc[:, 0].to_numpy() > pcutoff).astype(int),
                    dtype=pl.Int64,
                ),
            },
        )

        df_pl = df_pl.with_columns(
            merge_bouts(df_pl.select(PRED).to_series(), min_window_frames).alias(PRED),
        )

        behaviour_df_ls.append(df_pl)
        logger.info("Completed %s classification.", behaviour_name)

    if len(behaviour_df_ls) == 0:
        return pl.DataFrame(schema=BEHAVIOUR_PREDICTED_SCHEMA)

    return pl.concat(behaviour_df_ls)


def _validate_bodypoint_match(
    feat_cfg: ExtractFeaturesConfig,
    model_config: ClassifyBehaviourConfig,
) -> None:
    """Validate that the model's bodypoint config matches the experiment's."""
    if set(feat_cfg.individuals) != set(model_config.individuals):
        msg = (
            f"Individual mismatch for '{model_config.behaviour_name}': "
            f"experiment={sorted(feat_cfg.individuals)}, "
            f"model={sorted(model_config.individuals)}"
        )
        raise ValueError(msg)
    if set(feat_cfg.bodyparts) != set(model_config.bodyparts):
        msg = (
            f"Bodypart mismatch for '{model_config.behaviour_name}': "
            f"experiment={sorted(feat_cfg.bodyparts)}, "
            f"model={sorted(model_config.bodyparts)}"
        )
        raise ValueError(msg)
