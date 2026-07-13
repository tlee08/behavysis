"""Classify Behaviours."""

import numpy as np
import polars as pl
from loguru import logger

from behavysis.behaviour_classifier import ClassifierContract, predict_df
from behavysis.behaviour_classifier.storage import ClassifierFp
from behavysis.constants import PRED
from behavysis.models import ExperimentConfig, ExperimentMetadata, ExtractFeaturesConfig
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
        clf_proj = ClassifierFp(model_config.clf_fp)
        contract = ClassifierContract.read_yaml(clf_proj.contract_fp())
        _validate_feature_contract(feat_cfg, contract)

        min_window_secs = model_config.min_empty_window_secs
        min_window_frames = int(np.round(min_window_secs * metadata.require_fps()))

        behaviour_df_i = predict_df(model_config.clf_fp, features_df)

        behaviour_df_i = behaviour_df_i.with_columns(
            merge_bouts(
                behaviour_df_i.select(PRED).to_series(), min_window_frames
            ).alias(PRED),
        )

        behaviour_df_ls.append(behaviour_df_i)
        logger.info("Completed {} classification.", contract.behaviour_name)

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
