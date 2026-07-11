"""Classify Behaviours."""

import numpy as np
import polars as pl
from loguru import logger

from behavysis.behaviour_classifier import (
    BehaviourClassifier,
    ClassifierContract,
)
from behavysis.constants import PRED, PROB
from behavysis.models import (
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

    Each classifier's identity and feature contract are resolved from its
    ``production.yaml``. The experiment's ``extract_features`` config is
    validated against that contract before classifying.
    """
    model_config_ls = config.require_classify_behaviour()
    feat_cfg = config.require_extract_features()

    behaviour_df_ls = []
    for model_config in model_config_ls:
        behaviour_model = BehaviourClassifier.load(model_config.clf_fp.parent)
        _validate_feature_contract(feat_cfg, behaviour_model.contract)

        behaviour_name = behaviour_model.contract.behaviour_name
        pcutoff = model_config.pcutoff or behaviour_model.config.pcutoff
        min_window_secs = model_config.min_empty_window_secs
        min_window_frames = int(np.round(min_window_secs * metadata.require_fps()))

        behaviour_df_i = behaviour_model.predict(features_df)

        df_pl = behaviour_df_i.with_columns(
            (pl.col(PROB) > pcutoff).cast(pl.Int64).alias(PRED),
        )
        df_pl = df_pl.with_columns(
            merge_bouts(df_pl.select(PRED).to_series(), min_window_frames).alias(PRED),
        )

        behaviour_df_ls.append(df_pl)
        logger.info("Completed {} classification.", behaviour_name)

    if len(behaviour_df_ls) == 0:
        return pl.DataFrame(schema=BEHAVIOUR_PREDICTED_SCHEMA)

    return pl.concat(behaviour_df_ls)


def _validate_feature_contract(
    feat_cfg: ExtractFeaturesConfig,
    contract: ClassifierContract,
) -> None:
    """Validate the experiment's features match the classifier's contract."""
    if set(feat_cfg.individuals) != set(contract.individuals):
        msg = (
            f"Individual mismatch for '{contract.behaviour_name}': "
            f"experiment={sorted(feat_cfg.individuals)}, "
            f"model={sorted(contract.individuals)}"
        )
        raise ValueError(msg)
    if set(feat_cfg.bodyparts) != set(contract.bodyparts):
        msg = (
            f"Bodypart mismatch for '{contract.behaviour_name}': "
            f"experiment={sorted(feat_cfg.bodyparts)}, "
            f"model={sorted(contract.bodyparts)}"
        )
        raise ValueError(msg)
