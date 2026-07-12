"""Behaviour classifier — training and inference.

A classifier is fully self-contained in its own directory (``clf_dir``).
Training data lives inside it.  Each training run produces a numbered
iteration directory containing ``config.yaml``, ``model.joblib``, and
an ``evaluation/`` folder.  See ``storage`` for the full on-disk layout.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import joblib
import numpy as np
import polars as pl
from loguru import logger
from sklearn.metrics import classification_report

from behavysis.constants import (
    ACTUAL,
    BEHAVIOUR,
    EXPERIMENT,
    FRAME,
    PRED,
    PROB,
)
from behavysis.schemas import BEHAVIOUR_PREDICTED_SCHEMA

from .adapter import BaseAdapter, SklearnAdapter
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

    contract = ClassifierContract.read_yaml(clf_proj.contract_fp())

    iteration = _next_iteration(clf_proj.root_dir(), model_name)
    md = clf_proj.model_dir(model_name, iteration)

    config = TrainingRecipe(name=model_name, hyperparameters=hyperparameters)
    config.write_yaml(md / "config.yaml")

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
    adapter: BaseAdapter = factory()
    adapter.fit(train_df, config)

    # Save model
    adapter.save(md)

    # Evaluate
    ed = clf_proj.eval_dir(model_name, iteration)
    ed.mkdir(parents=True, exist_ok=True)
    _eval_split(adapter, train_df, config, ed, "train")
    _eval_split(adapter, test_df, config, ed, "test")

    # Diagnostics
    _run_diagnostics(adapter, clf_proj.root_dir(), ed)

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
    features_df: pl.DataFrame,
    pcutoff: float | None = None,
) -> pl.DataFrame:
    """Run inference on a wide features DataFrame.

    ``features_df`` has a ``frame`` column plus feature columns.
    Returns a long-form DataFrame with ``(frame, behaviour, prob, pred)``.
    """
    clf_proj = ClassifierFp(clf_contract_fp.parent)
    contract = ClassifierContract.read_yaml(clf_proj.contract_fp())
    config = TrainingRecipe.read_yaml(clf_proj.active_config_fp())
    pipeline = joblib.load(clf_proj.active_model_dir() / "model.joblib")
    pcutoff = pcutoff or config.pcutoff

    frames = features_df.get_column(FRAME)
    prob = pipeline.predict_proba(features_df.drop(FRAME).to_numpy())[:, 1]
    return pl.DataFrame(
        {
            FRAME: frames,
            BEHAVIOUR: [contract.behaviour_name] * len(frames),
            PROB: prob,
            PRED: (prob > pcutoff).astype(int),
        },
        schema=BEHAVIOUR_PREDICTED_SCHEMA,
    )


# ── internal helpers ─────────────────────────────────────────────────


def _eval_split(
    adapter: BaseAdapter,
    df: pl.DataFrame,
    config: TrainingRecipe,
    ed: Path,
    name: str,
) -> tuple[float | None, float | None]:
    y_true = df[ACTUAL].to_numpy()
    x = df.drop([EXPERIMENT, FRAME, ACTUAL]).to_numpy()
    y_prob = adapter.predict(x)
    y_pred = (y_prob > config.pcutoff).astype(int)

    # Save raw eval parquet
    df.select([EXPERIMENT, FRAME]).with_columns(
        pl.Series("y_true", y_true),
        pl.Series("y_prob", y_prob),
        pl.Series("y_pred", y_pred),
    ).write_parquet(ed / f"{name}_eval.parquet")

    report = classification_report(
        y_true,
        y_pred,
        target_names=["nil", "behav"],
        output_dict=True,
    )
    return report.get("accuracy"), report["behav"]["f1-score"]


def _run_diagnostics(
    adapter: BaseAdapter,
    clf_dir: Path,
    ed: Path,
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
        save_feature_importance(feature_names, importances, ed)
        save_feature_report(feature_names, importances, ed, n_features_total)
