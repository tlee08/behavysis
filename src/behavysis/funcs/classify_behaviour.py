"""Classify Behaviours."""

from pathlib import Path

import polars as pl

from behavysis.behaviour_classifier import predict


def classify_behaviour(
    contract_fp: Path,
    features_df: pl.DataFrame,
) -> pl.DataFrame:
    """Run inference with a single classifier on the given features DataFrame.

    Parameters
    ----------
    contract_fp : Path
        Path to the classifier contract YAML file.
    features_df : pl.DataFrame
        Wide features DataFrame with ``frame`` + feature columns.

    Returns:
    -------
    pl.DataFrame
        ``(frame, behaviour, prob, pred)`` long-form predictions.
    """
    return predict(contract_fp, features_df)
