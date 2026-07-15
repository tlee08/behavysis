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
import yaml
from loguru import logger

from behavysis.behaviour_classifier import MODEL_REGISTRY
from behavysis.constants import ACTUAL, EXPERIMENT

from .adapter import MODEL_TYPES_TO_CLASS, MODEL_TYPES_TO_STRING, BaseAdapter
from .config import ClassifierActive, ClassifierContract, TrainingRecipe
from .data import label_bouts, load_all_data, stratified_split_by_group
from .evaluation import EvalReport, make_eval_report
from .registry import ROUTINE_MODELS
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


# ── initialising ─────────────────────────────────────────────────────


def init_classifier(
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


# ── training ─────────────────────────────────────────────────────────


def train_model(
    contract_fp: Path,
    model_name: str,
    *,
    iteration: int | None = None,
    overwrite: bool = False,
    factory: Callable[[Path], BaseAdapter] | None = None,
) -> Path:
    """Train.

    Args:
        contract_fp (Path):
            Roots to the classifier project directory.
        model_name (str):
            model name (also used for `MODEL_REGISTRY` if factory not given).
        iteration (int | None, optional):
            If given then use, otherwise auto-next-iteration.
        overwrite (bool, optional):
            Allow retraining an existing iteration. Defaults to False.
        factory (Callable[[Path], BaseAdapter] | None, optional):
            Override registry. Defaults to None.

    Returns:
        Path: `model_dir` path.
    """
    # Define project files
    clf_proj = ClassifierFp(contract_fp.parent)
    clf_dir = clf_proj.root_dir()
    contract_fp = clf_proj.contract_fp()
    iteration = iteration or _next_iteration(clf_dir, model_name)
    model_dir = clf_proj.model_dir(model_name, iteration)
    eval_dir = clf_proj.eval_dir(model_name, iteration)
    config_fp = clf_proj.config_fp(model_name, iteration)
    # Get contract data
    contract = ClassifierContract.read_yaml(contract_fp)

    # Load adapter
    factory = factory or MODEL_REGISTRY[model_name]
    adapter = factory(config_fp)

    # Check overwrite and make config
    if config_fp.exists():
        if not overwrite:
            return model_dir
        config = TrainingRecipe(
            behaviour_name=contract.behaviour_name,
            model_name=model_name,
            iteration=iteration,
            model_type=MODEL_TYPES_TO_STRING[type(adapter)],
        )
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
    train_df = df[train_idx]
    test_df = df[test_idx]

    # Train
    logger.info("Training {} (iteration={:03d})", contract.behaviour_name, iteration)
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

    logger.info(
        "Training complete: {} {:03d}",
        model_name,
        iteration,
    )
    return model_dir


def train_all_models(contract_fp: Path) -> list[Path]:
    """Train the routine model set, one iteration each."""
    return [train_model(contract_fp, name) for name in ROUTINE_MODELS]


# ── set best model ───────────────────────────────────────────────────


def promote_best(contract_fp: Path) -> ClassifierActive:
    """Scan all iterations, pick best by F2 on bout-level test, write active.yaml."""
    # Get configs
    clf_proj = ClassifierFp(contract_fp.parent)
    models_dir = clf_proj.models_dir()
    contract = ClassifierContract.read_yaml(clf_proj.contract_fp())
    # Get metrics from each model
    pattern = re.compile(r"^(.+)-(\d+)$")
    candidates: list[tuple[str, int, float]] = []  # (name, iter, eval_metric_value)
    for subdir in models_dir.iterdir():
        m = pattern.match(subdir.name)
        if not m:
            continue
        model_name, iteration = m.group(1), int(m.group(2))
        eval_report_fp = clf_proj.eval_dir(model_name, iteration) / "bouts_report.yaml"
        if not eval_report_fp.exists():
            logger.warning(
                "No bouts_report.yaml for {}-{:03d}, skipping", model_name, iteration
            )
            continue
        eval_report = yaml.safe_load(eval_report_fp.read_text())
        # Actually save_eval_report uses yaml.dump, so read with yaml
        if "test" not in eval_report:
            logger.warning(
                "No test split in bouts_report for {}-{:03d}, skipping",
                model_name,
                iteration,
            )
            continue
        metrics = eval_report["test"]
        if contract.eval_metric not in metrics:
            continue
        candidates.append((model_name, iteration, metrics[contract.eval_metric]))
    # Choose best metric
    if not candidates:
        msg = f"No valid model iterations found in {models_dir}"
        raise FileNotFoundError(msg)
    if contract.eval_metric_higher_better:
        best = max(candidates, key=lambda c: c[2])
    else:
        best = min(candidates, key=lambda c: c[2])
    # Set best metric
    active = ClassifierActive(model_name=best[0], iteration=best[1])
    active.write_yaml(clf_proj.active_fp())
    logger.info(
        "Promoted {}-{:03d} ({}={:.4f})",
        best[0],
        best[1],
        contract.eval_metric,
        best[2],
    )
    return active


# ── inference ────────────────────────────────────────────────────────


def predict_choose_model(
    contract_fp: Path,
    model_name: str,
    iteration: int,
    x_df: pl.DataFrame,
) -> pl.DataFrame:
    """Run inference on a wide features DataFrame.

    ``features_df`` has a ``frame`` column plus feature columns.
    Returns a long-form DataFrame with ``(frame, behaviour, prob, pred)``.
    """
    # Configs
    clf_proj = ClassifierFp(contract_fp.parent)
    config_fp = clf_proj.config_fp(model_name, iteration)
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
    return predict_choose_model(contract_fp, active.model_name, active.iteration, x_df)


# ── other helpers ─────────────────────────────────────────────────


def make_eval_report_choose_model(
    contract_fp: Path, model_name: str, iteration: int
) -> EvalReport:
    """Run make_eval_report by giving a model's filepath."""
    # Get filepaths
    clf_proj = ClassifierFp(contract_fp.parent)
    eval_dir = clf_proj.eval_dir(model_name, iteration)
    # Read raw eval data
    eval_train_df = pl.read_parquet(eval_dir / "train_eval.parquet")
    eval_test_df = pl.read_parquet(eval_dir / "test_eval.parquet")
    # Make evaluation
    return make_eval_report({"train": eval_train_df, "test": eval_test_df})
