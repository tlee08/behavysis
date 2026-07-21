"""Classify Behaviours."""

from pathlib import Path

import polars as pl
from loguru import logger

from behavysis.behaviour_classifier import ClassifierContract, predict
from behavysis.behaviour_classifier.storage import ClassifierFp


def classify_single(
    contract_fp: Path,
    features_df: pl.DataFrame,
) -> pl.DataFrame:
    """Run inference with a single classifier on the given features DataFrame.

    Parameters
    ----------
    contract_fp : Path
        Path to the classifier's contract.yaml.
    features_df : pl.DataFrame
        Wide features DataFrame with ``frame`` + feature columns.

    Returns:
    -------
    pl.DataFrame
        ``(frame, behaviour, prob, pred)`` long-form predictions.
    """
    clf_proj = ClassifierFp(contract_fp.parent)
    contract = ClassifierContract.read_yaml(clf_proj.contract_fp())
    result = predict(contract_fp, features_df)
    logger.info("Completed {} classification.", contract.behaviour_name)
    return result
