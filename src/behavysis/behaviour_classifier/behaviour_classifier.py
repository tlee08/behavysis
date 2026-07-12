"""Behaviour classifier — training and inference.

A classifier is fully self-contained in its own directory (``clf_dir``).
Training data lives inside it.  Each training run produces a numbered
iteration directory containing ``config.yaml``, ``model.joblib``, and
an ``evaluation/`` folder.  See ``storage`` for the full on-disk layout.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from loguru import logger
from sklearn.metrics import classification_report

from behavysis.constants import (
    ACTUAL,
    BEHAVIOUR,
    EXPERIMENT,
    PRED,
    PROB,
)
from behavysis.schemas import BEHAVIOUR_PREDICTED_SCHEMA

from .adapter import (
    MODEL_TYPES_TO_CLASS,
    MODEL_TYPES_TO_STRING,
    BaseAdapter,
    SklearnAdapter,
)
from .config import ClassifierContract, TrainingRecipe
from .data import load_feature_names, load_training_data, stratified_split_by_bout
from .evaluation import save_feature_importance, save_feature_report
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
    _eval_split(adapter, train_df, config, eval_dir, "train")
    _eval_split(adapter, test_df, config, eval_dir, "test")

    # Diagnostics
    _run_diagnostics(adapter, clf_proj.root_dir(), eval_dir)

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
    contract = ClassifierContract.read_yaml(clf_proj.contract_fp())
    config = TrainingRecipe.read_yaml(clf_proj.active_config_fp())
    model = MODEL_TYPES_TO_CLASS[config.model_type].load(clf_proj.active_model_dir())
    pcutoff = pcutoff or config.pcutoff

    prob_df = model.predict(x_df)
    prob_df = prob_df.with_columns(
        pl.lit(contract.behaviour_name).alias(BEHAVIOUR),
        (pl.col(PROB) > pcutoff).cast(pl.Int64).alias(PRED),
    )
    return pl.DataFrame(prob_df, schema=BEHAVIOUR_PREDICTED_SCHEMA)


# ── internal helpers ─────────────────────────────────────────────────


def _eval_split(
    adapter: BaseAdapter,
    df: pl.DataFrame,
    config: TrainingRecipe,
    eval_dir: Path,
    name: str,
) -> tuple[float | None, float | None]:
    x_df = df.drop([EXPERIMENT, ACTUAL])
    y_df = adapter.predict(x_df)
    y_df = y_df.with_columns(
        df[EXPERIMENT].alias(EXPERIMENT),
        df[ACTUAL].alias(ACTUAL),
        (pl.col(PROB) > config.pcutoff).cast(pl.Int64).alias(PRED),
    )
    # Save raw eval df to parquet
    y_df.write_parquet(eval_dir / f"{name}_eval.parquet")

    report = classification_report(
        y_df[ACTUAL],
        y_df[PRED],
        target_names=["nil", "behav"],
        output_dict=True,
    )
    return report.get("accuracy"), report["behav"]["f1-score"]


def _run_diagnostics(
    adapter: BaseAdapter,
    clf_dir: Path,
    eval_dir: Path,
) -> None:
    clf_proj = ClassifierFp(clf_dir)
    feature_names = load_feature_names(clf_proj.features_dir())
    if not feature_names:
        logger.warning("No feature names found for diagnostics.")
        return

    n_features_total = len(feature_names)

    importances: np.ndarray | None = None
    if isinstance(adapter, SklearnAdapter):
        named = adapter.pipeline.named_steps
        mask = np.arange(len(feature_names))
        if "selector" in named:
            mask = mask[named["selector"].get_support()]
        elif "var_filter" in named:
            mask = mask[named["var_filter"].get_support()]
        feature_names = [feature_names[i] for i in mask]
        importances = np.zeros(len(feature_names), dtype=np.float64)
        est = named["clf"]
        if hasattr(est, "feature_importances_"):
            importances = est.feature_importances_
        elif hasattr(est, "coef_"):
            importances = np.abs(est.coef_).flatten()

    if importances is not None:
        save_feature_importance(feature_names, importances, eval_dir)
        save_feature_report(feature_names, importances, eval_dir, n_features_total)
