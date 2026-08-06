"""Behaviour classifier — training and inference.

A classifier is fully self-contained in its own directory (``clf_dir``).
Training data lives inside it.  Each training run produces a model
directory containing ``recipe.yaml``, ``model.joblib``, and
an ``evaluation/`` folder.  See ``storage`` for the full on-disk layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import yaml
from loguru import logger

from behavysis.constants import ACTUAL, BOUT, EXPERIMENT, FRAME
from behavysis.utils import clean_memory, pass_exception, trace

from .adapter import MODEL_TYPES_TO_CLASS, MODEL_TYPES_TO_STRING, BaseAdapter
from .config import ActiveModel, ClassifierContract, ModelRecipe
from .data import label_bouts, load_all_data, stratified_split_by_group
from .evaluation import EvalResult, make_eval_result
from .registry import MODEL_REGISTRY
from .storage import ClassifierPaths

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


# -- initialising -----------------------------------------------------


def write_contract(
    contract_fp: Path,
    behaviour_name: str,
    training_project_path: Path,
    feature_set: str,
    *,
    overwrite: bool,
) -> ClassifierPaths:
    """Make contract for classifier."""
    if not contract_fp.exists() or overwrite:
        contract = ClassifierContract(
            behaviour_name=behaviour_name,
            training_project_path=training_project_path,
            feature_set=feature_set,
        )
        contract.write_yaml(contract_fp)
    return ClassifierPaths(contract_fp)


# -- model discovery -----------------------------------------------


def list_models(contract_fp: Path) -> list[str]:
    """List all models."""
    clf = ClassifierPaths(contract_fp)
    if not clf.models_dir().exists():
        return []
    return [
        _i.stem for _i in clf.models_dir().iterdir() if clf.recipe_fp(_i.stem).exists()
    ]


# -- training ---------------------------------------------------------


@clean_memory
def train_model(
    contract_fp: Path,
    model_name: str,
    *,
    overwrite: bool,
    factory: Callable[[Path], BaseAdapter] | None = None,
) -> Path:
    """Train.

    Args:
        contract_fp (Path):
            Path to the classifier contract YAML file.
        model_name (str):
            model name (also used for `MODEL_REGISTRY` if factory not given).
        overwrite (bool, optional):
            Allow retraining an existing model. Defaults to False.
        factory (Callable[[Path], BaseAdapter] | None, optional):
            Override registry. Defaults to None.

    Returns:
        Path: `model_dir` path.
    """
    # Define paths
    clf = ClassifierPaths(contract_fp)
    model_dir = clf.model_dir(model_name)
    eval_dir = clf.eval_dir(model_name)
    recipe_fp = clf.recipe_fp(model_name)
    # Check overwrite
    if eval_dir.exists() and not overwrite:
        return model_dir
    # Load adapter
    factory = factory or MODEL_REGISTRY[model_name]
    adapter = factory(recipe_fp)
    # Write recipe if not exists
    if not recipe_fp.exists():
        ModelRecipe(
            behaviour_name=clf.contract().behaviour_name,
            model_name=model_name,
            model_type=MODEL_TYPES_TO_STRING[type(adapter)],
        ).write_yaml(recipe_fp)
    # Get recipe data
    recipe = ModelRecipe.read_yaml(recipe_fp)

    # Load and align data
    df = load_all_data(
        clf.features_dir(), clf.labels_dir(), clf.contract().behaviour_name
    )
    df = label_bouts(df, ACTUAL)

    # Split into train / test (experiment-level grouping)
    train_idx, test_idx = stratified_split_by_group(
        df, recipe.test_split, EXPERIMENT, recipe.seed
    )
    train_df = df.gather(train_idx).sort([EXPERIMENT, FRAME])
    test_df = df.gather(test_idx).sort([EXPERIMENT, FRAME])

    # Train
    logger.info("Training {}", clf.contract().behaviour_name)
    adapter.fit(train_df)

    # Save model
    adapter.save()

    # Load model from save
    adapter = adapter.load(recipe_fp)

    # Evaluate
    eval_dir.mkdir(parents=True, exist_ok=True)
    # Predictions
    adapter.predict(train_df).with_columns(
        train_df.get_column(EXPERIMENT), train_df.get_column(ACTUAL)
    ).write_parquet(eval_dir / "train_eval.parquet")
    adapter.predict(test_df).with_columns(
        test_df.get_column(EXPERIMENT), test_df.get_column(ACTUAL)
    ).write_parquet(eval_dir / "test_eval.parquet")
    # Further evaluation
    res = make_eval_result_choose_model(clf.contract_fp(), model_name)
    # Save. Only report and charts, not df
    for _name, _report in res["report"].items():
        (eval_dir / f"{_name}.yaml").write_text(yaml.dump(_report))
    for _name, _chart in res["chart"].items():
        _chart.save(eval_dir / f"{_name}.png")

    logger.info("Training complete: {}", model_name)
    return model_dir


def train_all_models(contract_fp: Path, *, overwrite: bool) -> list[Path]:
    """Train the routine model set."""
    return [
        pass_exception(trace(train_model))(
            contract_fp=contract_fp,
            model_name=model_name,
            overwrite=overwrite,
        )
        for model_name in MODEL_REGISTRY
    ]


# -- set best model ---------------------------------------------------


def promote_best(contract_fp: Path) -> ActiveModel:
    """Promote the model with the best evaluation metric."""
    clf = ClassifierPaths(contract_fp)
    # Get eval_metric values for each model
    scores = []
    for model_name in list_models(contract_fp):
        report_fp = clf.eval_dir(model_name) / f"{BOUT}_report.yaml"
        if not report_fp.exists():
            continue
        report = yaml.safe_load(report_fp.read_text())
        score = report.get("test", {}).get(clf.contract().eval_metric)
        if score is not None:
            scores.append((model_name, score))
    # Get best model, given the list of eval_metric scores
    if not scores:
        msg = f"No valid model evaluations found in {clf.models_dir()}"
        raise FileNotFoundError(msg)
    model_name, score = (
        max(scores, key=lambda x: x[1])
        if clf.contract().eval_metric_higher_better
        else min(scores, key=lambda x: x[1])
    )
    # Update active.yaml
    active = ActiveModel(model_name=model_name)
    active.write_yaml(clf.active_fp())
    logger.info(f"Promoted {model_name} ({clf.contract().eval_metric}={score})")
    return active


# -- retrieve model --------------------------------------------------------


def load_adapter(contract_fp: Path, model_name: str) -> BaseAdapter:
    """Load adapter, given classifier layout and model name."""
    clf = ClassifierPaths(contract_fp)
    recipe_fp = clf.recipe_fp(model_name)
    # Check if model exists
    if not recipe_fp.exists():
        msg = "Classifier's model not found. Check active.yaml."
        raise ValueError(msg)
    # Recipe
    recipe = ModelRecipe.read_yaml(recipe_fp)
    # Load model
    return MODEL_TYPES_TO_CLASS[recipe.model_type].load(recipe_fp)


# -- inference --------------------------------------------------------


@clean_memory
def predict_choose_model(
    contract_fp: Path, model_name: str, x_df: pl.DataFrame
) -> pl.DataFrame:
    """Run inference with a specific model on a wide features DataFrame.

    ``x_df`` has a ``frame`` column plus feature columns.
    Returns a long-form DataFrame with ``(frame, behaviour, prob, pred)``.
    """
    adapter = load_adapter(contract_fp, model_name)
    return adapter.predict(x_df)


def predict(contract_fp: Path, x_df: pl.DataFrame) -> pl.DataFrame:
    """Run inference with the active model on a wide features DataFrame.

    ``x_df`` has a ``frame`` column plus feature columns.
    Returns a long-form DataFrame with ``(frame, behaviour, prob, pred)``.
    """
    clf = ClassifierPaths(contract_fp)
    active = ActiveModel.read_yaml(clf.active_fp())
    return predict_choose_model(contract_fp, active.model_name, x_df)


# -- other helpers -------------------------------------------------


def make_eval_result_choose_model(contract_fp: Path, model_name: str) -> EvalResult:
    """Run make_eval_report by giving a model's filepath."""
    clf = ClassifierPaths(contract_fp)
    eval_dir = clf.eval_dir(model_name)
    return make_eval_result(
        {
            "train": pl.read_parquet(eval_dir / "train_eval.parquet"),
            "test": pl.read_parquet(eval_dir / "test_eval.parquet"),
        }
    )
