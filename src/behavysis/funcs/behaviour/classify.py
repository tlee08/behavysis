"""Classify Behaviours."""

import polars as pl
from loguru import logger

from behavysis.behaviour_classifier import ClassifierContract, ClassifierPaths, predict


def classify_single(
    clf: ClassifierPaths,
    features_df: pl.DataFrame,
) -> pl.DataFrame:
    """Run inference with a single classifier on the given features DataFrame.

    Parameters
    ----------
    clf : ClassifierPaths
        The classifier's directory layout helper.
    features_df : pl.DataFrame
        Wide features DataFrame with ``frame`` + feature columns.

    Returns:
    -------
    pl.DataFrame
        ``(frame, behaviour, prob, pred)`` long-form predictions.
    """
    contract = ClassifierContract.read_yaml(clf.contract_fp())
    result = predict(clf, features_df)
    logger.info("Completed {} classification.", contract.behaviour_name)
    return result
