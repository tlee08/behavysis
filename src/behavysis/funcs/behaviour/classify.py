"""Classify Behaviours."""

import polars as pl
from loguru import logger

from behavysis.behaviour_classifier import ClassifierContract, predict
from behavysis.behaviour_classifier.storage import ClassifierFp
from behavysis.models import ExperimentConfig, ExperimentMetadata
from behavysis.schemas import BEHAVIOUR_PREDICTED_SCHEMA


def classify_behaviour(
    features_df: pl.DataFrame,
    config: ExperimentConfig,
    metadata: ExperimentMetadata,  # noqa: ARG001
) -> pl.DataFrame:
    """Classify behaviour using trained models.

    Each classifier's identity and feature contract are resolved from its
    ``contract.yaml``. The experiment's ``extract_features`` config is
    validated against that contract before classifying.
    """
    model_config_ls = config.require_classify_behaviour()

    behaviour_df_ls = []
    for model_config in model_config_ls:
        clf_proj = ClassifierFp(model_config.contract_fp.parent)
        contract = ClassifierContract.read_yaml(clf_proj.contract_fp())

        behaviour_df_i = predict(model_config.contract_fp, features_df)

        behaviour_df_ls.append(behaviour_df_i)
        logger.info("Completed {} classification.", contract.behaviour_name)

    if len(behaviour_df_ls) == 0:
        return pl.DataFrame(schema=BEHAVIOUR_PREDICTED_SCHEMA)

    return pl.concat(behaviour_df_ls)
