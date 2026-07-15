"""Behaviour classifier — training and inference.

A classifier is fully self-contained in its own directory (``clf_dir``).
Training data lives inside it.  Each training run produces a numbered
directory containing ``config.yaml``, ``model.joblib``, and
an ``evaluation/`` folder.  See ``storage`` for the full on-disk layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import yaml
from loguru import logger

from behavysis.constants import ACTUAL, BOUT, EXPERIMENT, FRAME
from behavysis.utils import pass_exception, trace

from .adapter import MODEL_TYPES_TO_CLASS, MODEL_TYPES_TO_STRING, BaseAdapter
from .config import ClassifierActive, ClassifierContract, TrainingRecipe
from .data import label_bouts, load_all_data, stratified_split_by_group
from .evaluation import EvalResult, make_eval_report
from .registry import MODEL_REGISTRY
from .storage import ClassifierFp

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


# ── initialising ─────────────────────────────────────────────────────


def write_contract(
    contract_fp: Path,
    behaviour_name: str,
    individuals: list[str],
    bodyparts: list[str],
    *,
    overwrite: bool = False,
) -> ClassifierContract:
    """Make contract for classifier."""
    # Write
    if not contract_fp.exists() or overwrite:
        contract = ClassifierContract(
            behaviour_name=behaviour_name,
            individuals=individuals,
            bodyparts=bodyparts,
        )
        contract.write_yaml(contract_fp)
    # Read
    return ClassifierContract.read_yaml(contract_fp)


# ── model discovery ───────────────────────────────────────────────


def list_models(contract_fp: Path) -> list[str]:
    """List all models."""
    clf_proj = ClassifierFp(contract_fp.parent)
    if not clf_proj.models_dir().exists():
        return []
    return [
        _i.stem
        for _i in clf_proj.models_dir().iterdir()
        if clf_proj.config_fp(_i.stem).exists()
    ]


# ── training ─────────────────────────────────────────────────────────


def train_model(
    contract_fp: Path,
    model_name: str,
    *,
    overwrite: bool = False,
    factory: Callable[[Path], BaseAdapter] | None = None,
) -> Path:
    """Train.

    Args:
        contract_fp (Path):
            The classifier project's contract file.
        model_name (str):
            model name (also used for `MODEL_REGISTRY` if factory not given).
        overwrite (bool, optional):
            Allow retraining an existing model. Defaults to False.
        factory (Callable[[Path], BaseAdapter] | None, optional):
            Override registry. Defaults to None.

    Returns:
        Path: `model_dir` path.
    """
    # Define project files
    clf_proj = ClassifierFp(contract_fp.parent)
    contract_fp = clf_proj.contract_fp()
    model_dir = clf_proj.model_dir(model_name)
    eval_dir = clf_proj.eval_dir(model_name)
    config_fp = clf_proj.config_fp(model_name)
    # Get contract data
    contract = ClassifierContract.read_yaml(contract_fp)
    # Check overwrite
    if eval_dir.exists() and not overwrite:
        return model_dir
    # Load adapter
    factory = factory or MODEL_REGISTRY[model_name]
    adapter = factory(config_fp)
    # Make config
    if not config_fp.exists():
        TrainingRecipe(
            behaviour_name=contract.behaviour_name,
            model_name=model_name,
            model_type=MODEL_TYPES_TO_STRING[type(adapter)],
        ).write_yaml(config_fp)
    # Get config data
    config = TrainingRecipe.read_yaml(config_fp)

    # Load and align data
    df = load_all_data(
        clf_proj.features_dir(),
        clf_proj.labels_dir(),
        contract.behaviour_name,
    )
    df = label_bouts(df)

    # Split into train / test (experiment-level grouping)
    train_idx, test_idx = stratified_split_by_group(
        df, config.test_split, EXPERIMENT, config.seed
    )
    train_df = df[train_idx].sort([EXPERIMENT, FRAME])
    test_df = df[test_idx].sort([EXPERIMENT, FRAME])

    # Train
    logger.info("Training {}", contract.behaviour_name)
    adapter.fit(train_df)

    # Save model
    adapter.save()

    # Load model from save
    adapter = adapter.load(config_fp)

    # Evaluate
    eval_dir.mkdir(parents=True, exist_ok=True)
    # Predictions
    eval_train_df = adapter.predict(train_df).with_columns(
        train_df[EXPERIMENT], train_df[ACTUAL]
    )
    eval_train_df.write_parquet(eval_dir / "train_eval.parquet")
    eval_test_df = adapter.predict(test_df).with_columns(
        test_df[EXPERIMENT], test_df[ACTUAL]
    )
    eval_test_df.write_parquet(eval_dir / "test_eval.parquet")
    # Further evaluation
    res = make_eval_report({"train": eval_train_df, "test": eval_test_df})
    # Save. Only report and charts, not df
    for _name, _report in res["report"].items():
        (eval_dir / f"{_name}.yaml").write_text(yaml.dump(_report))
    for _name, _chart in res["chart"].items():
        _chart.save(eval_dir / f"{_name}.png")

    logger.info("Training complete: {}", model_name)
    return model_dir


def train_all_models(contract_fp: Path) -> list[Path]:
    """Train the routine model set."""
    return [
        pass_exception(trace(train_model))(contract_fp, name) for name in MODEL_REGISTRY
    ]


# ── set best model ───────────────────────────────────────────────────


def promote_best(contract_fp: Path) -> ClassifierActive:
    """Promote the model with the best evaluation metric."""
    clf_proj = ClassifierFp(contract_fp.parent)
    contract = ClassifierContract.read_yaml(clf_proj.contract_fp())
    # Get eval_metric values for each model
    scores = []
    for model_name in list_models(contract_fp):
        report_fp = clf_proj.eval_dir(model_name) / f"{BOUT}_report.yaml"
        if not report_fp.exists():
            continue
        report = yaml.safe_load(report_fp.read_text())
        score = report.get("test", {}).get(contract.eval_metric)
        if score is not None:
            scores.append((model_name, score))
    # Get best model, given the list of eval_metric scores
    if not scores:
        msg = f"No valid model evaluations found in {clf_proj.models_dir()}"
        raise FileNotFoundError(msg)
    model_name, score = (
        max(scores, key=lambda x: x[1])
        if contract.eval_metric_higher_better
        else min(scores, key=lambda x: x[1])
    )
    # Update active.yaml
    active = ClassifierActive(model_name=model_name)
    active.write_yaml(clf_proj.active_fp())
    logger.info(f"Promoted {model_name} ({contract.eval_metric}={score})")
    return active


# ── inference ────────────────────────────────────────────────────────


def predict_choose_model(
    contract_fp: Path,
    model_name: str,
    x_df: pl.DataFrame,
) -> pl.DataFrame:
    """Run inference on a wide features DataFrame.

    ``features_df`` has a ``frame`` column plus feature columns.
    Returns a long-form DataFrame with ``(frame, behaviour, prob, pred)``.
    """
    clf_proj = ClassifierFp(contract_fp.parent)
    config_fp = clf_proj.config_fp(model_name)
    # Check if model exists
    if not config_fp.exists():
        msg = "Classifier's model not found. Check active.yaml."
        raise ValueError(msg)
    # Configs
    config = TrainingRecipe.read_yaml(config_fp)
    # Load model
    adapter = MODEL_TYPES_TO_CLASS[config.model_type].load(config_fp)
    # Run inference
    return adapter.predict(x_df)


def predict(
    contract_fp: Path,
    x_df: pl.DataFrame,
) -> pl.DataFrame:
    """Run inference on a wide features DataFrame.

    ``features_df`` has a ``frame`` column plus feature columns.
    Returns a long-form DataFrame with ``(frame, behaviour, prob, pred)``.
    """
    # Retrieve active model
    clf_proj = ClassifierFp(contract_fp.parent)
    active = ClassifierActive.read_yaml(clf_proj.active_fp())
    # Predict
    return predict_choose_model(contract_fp, active.model_name, x_df)


# ── other helpers ─────────────────────────────────────────────────


def make_eval_report_choose_model(contract_fp: Path, model_name: str) -> EvalResult:
    """Run make_eval_report by giving a model's filepath."""
    # Get filepaths
    clf_proj = ClassifierFp(contract_fp.parent)
    eval_dir = clf_proj.eval_dir(model_name)
    # Read raw eval data
    eval_train_df = pl.read_parquet(eval_dir / "train_eval.parquet")
    eval_test_df = pl.read_parquet(eval_dir / "test_eval.parquet")
    # Make evaluation
    return make_eval_report({"train": eval_train_df, "test": eval_test_df})
