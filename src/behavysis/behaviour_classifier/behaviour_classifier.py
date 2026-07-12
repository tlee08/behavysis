"""Behaviour classifier — training and inference.

A classifier is fully self-contained in its own directory (``clf_dir``).
Training data lives inside it.  Each training run produces a numbered
iteration directory containing ``config.yaml``, ``model.joblib``, and
an ``evaluation/`` folder.  See ``storage`` for the full on-disk layout.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import polars as pl
from loguru import logger

from behavysis.constants import ACTUAL, BEHAVIOUR, EXPERIMENT, PRED, PROB
from behavysis.schemas import BEHAVIOUR_PREDICTED_SCHEMA

from .adapter import MODEL_TYPES_TO_CLASS, MODEL_TYPES_TO_STRING, BaseAdapter
from .config import ClassifierActive, ClassifierContract, TrainingRecipe
from .data import load_training_data, stratified_split_by_bout
from .registry import MODEL_REGISTRY
from .storage import ClassifierFp

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


# ── iteration numbering ───────────────────────────────────────────────


def _next_iteration(clf_dir: Path, model_name: str) -> int:
    """Scan classifiers/ for {name}-NNN dirs, return max + 1."""
    clf_proj = ClassifierFp(clf_dir)
    cd = clf_proj.models_dir()
    if not cd.exists():
        return 1
    pattern = re.compile(rf"^{re.escape(model_name)}-(\d+)$")
    nums = [
        int(m.group(1))
        for d in cd.iterdir()
        if d.is_dir() and (m := pattern.match(d.name))
    ]
    return max(nums) + 1 if nums else 1


# ── training ─────────────────────────────────────────────────────────


def train(
    clf_contract_fp: Path,
    model_name: str,
    factory: Callable[[], BaseAdapter],
    hyperparameters: dict[str, list[object]],
) -> int:
    """Train a classifier and persist artifacts in a new iteration directory.

    Returns the iteration number.
    """
    clf_proj = ClassifierFp(clf_contract_fp.parent)
    contract_fp = clf_proj.contract_fp()
    iteration = _next_iteration(clf_proj.root_dir(), model_name)
    model_dir = clf_proj.model_dir(model_name, iteration)
    config_fp = clf_proj.config_fp(model_name, iteration)
    contract = ClassifierContract.read_yaml(contract_fp)
    adapter = factory()
    config = TrainingRecipe(
        model_name=model_name,
        model_type=MODEL_TYPES_TO_STRING[type(adapter)],
        hyperparameters=hyperparameters,
    )
    config.write_yaml(config_fp)

    logger.info(
        "Training {} (iteration={:03d})",
        contract.behaviour_name,
        iteration,
    )

    # Load and align data
    df = load_training_data(
        clf_proj.features_dir(),
        clf_proj.labels_dir(),
        contract.behaviour_name,
    )

    # Split into train / test (bout-level grouping)
    train_mask, test_mask = stratified_split_by_bout(
        df,
        config.test_split,
        config.seed,
    )
    train_df = df.filter(train_mask)
    test_df = df.filter(test_mask)

    # Train
    adapter.fit(train_df, config)

    # Save model
    adapter.save(model_dir)

    # Evaluate
    eval_dir = clf_proj.eval_dir(model_name, iteration)
    eval_dir.mkdir(parents=True, exist_ok=True)
    _eval_split(
        eval_dir, adapter, "train", train_df, contract.behaviour_name, config.pcutoff
    )
    _eval_split(
        eval_dir, adapter, "test", test_df, contract.behaviour_name, config.pcutoff
    )

    logger.info(
        "Training complete: {} {:03d}",
        model_name,
        iteration,
    )
    return iteration


def train_all_models(clf_contract_fp: Path) -> list[int]:
    """Train all model types in the registry, one iteration each."""
    results: list[int] = []
    for model_name in MODEL_REGISTRY:
        factory, hyperparameters = MODEL_REGISTRY[model_name]
        results.append(train(clf_contract_fp, model_name, factory, hyperparameters))
    return results


# ── inference ────────────────────────────────────────────────────────


def predict_df_from_adapter(
    adapter: BaseAdapter,
    x_df: pl.DataFrame,
    behaviour_name: str,
    pcutoff: float | None = None,
) -> pl.DataFrame:
    """Run inference on a wide features DataFrame.

    ``features_df`` has a ``frame`` column plus feature columns.
    Returns a long-form DataFrame with ``(frame, behaviour, prob, pred)``.
    """
    # Run inference
    prob_df = adapter.predict(x_df)
    prob_df = prob_df.with_columns(
        pl.lit(behaviour_name).alias(BEHAVIOUR),
        (pl.col(PROB) > pcutoff).cast(pl.Int64).alias(PRED),
    )
    return pl.DataFrame(prob_df, schema=BEHAVIOUR_PREDICTED_SCHEMA)


def predict_df_choose_model(
    clf_contract_fp: Path,
    model_name: str,
    iteration: int,
    x_df: pl.DataFrame,
    pcutoff: float | None = None,
) -> pl.DataFrame:
    """Run inference on a wide features DataFrame.

    ``features_df`` has a ``frame`` column plus feature columns.
    Returns a long-form DataFrame with ``(frame, behaviour, prob, pred)``.
    """
    # Configs
    clf_proj = ClassifierFp(clf_contract_fp.parent)
    model_dir = clf_proj.model_dir(model_name, iteration)
    contract = ClassifierContract.read_yaml(clf_proj.contract_fp())
    config = TrainingRecipe.read_yaml(clf_proj.config_fp(model_name, iteration))
    pcutoff = pcutoff or config.pcutoff
    # Load model
    adapter = MODEL_TYPES_TO_CLASS[config.model_type].load(model_dir)
    # Run inference
    return predict_df_from_adapter(adapter, x_df, contract.behaviour_name, pcutoff)


def predict_df(
    clf_contract_fp: Path,
    x_df: pl.DataFrame,
    pcutoff: float | None = None,
) -> pl.DataFrame:
    """Run inference on a wide features DataFrame.

    ``features_df`` has a ``frame`` column plus feature columns.
    Returns a long-form DataFrame with ``(frame, behaviour, prob, pred)``.
    """
    clf_proj = ClassifierFp(clf_contract_fp.parent)
    active = ClassifierActive.read_yaml(clf_proj.active_fp())
    return predict_df_choose_model(
        clf_contract_fp, active.model_name, active.iteration, x_df, pcutoff
    )


# ── internal helpers ─────────────────────────────────────────────────


def _eval_split(
    eval_dir: Path,
    adapter: BaseAdapter,
    subset_name: str,
    df: pl.DataFrame,
    behaviour_name: str,
    pcutoff: float | None = None,
) -> None:
    # Run inference
    x_df = df.drop([EXPERIMENT, ACTUAL])
    y_df = predict_df_from_adapter(adapter, x_df, behaviour_name, pcutoff)
    y_df = y_df.with_columns(
        df[EXPERIMENT].alias(EXPERIMENT),
        df[ACTUAL].alias(ACTUAL),
    )
    # Save raw eval df to parquet
    y_df.write_parquet(eval_dir / f"{subset_name}_eval.parquet")
