"""Shared helpers for behaviour-classifier EDA and experiments.

Every diagnostic script in this package reads the same local data layout and
writes results to ``data/front-rear/eda/``.  The helpers here keep that layout,
the experiment-name parsing, and the metric definitions in one place so the
individual scripts stay small and consistent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from behavysis.behaviour_classifier.data import ACTUAL, load_all_data
from behavysis.constants import BOUT_ID, EXPERIMENT, FRAME, PROB
from behavysis.transforms import label_bouts

if TYPE_CHECKING:
    from collections.abc import Sequence

# -- local data layout ----------------------------------------------------

CLF_ROOT = Path("data/front-rear")
FEATURES_DIR = CLF_ROOT / "training_data" / "5_features_extracted" / "extract_rearing"
LABELS_DIR = CLF_ROOT / "training_data" / "7_behaviour_scored"
MODELS_DIR = CLF_ROOT / "models"
EDA_OUT_DIR = CLF_ROOT / "eda"

BEHAVIOUR = "FR"

MODELS = ["xgb", "tabpfn"]

# Parsed experiment-name columns (e.g. "4M1-1B-HOT-a").
ANIMAL = "animal"
BATCH = "batch"
CONDITION = "condition"
CAMERA = "camera"

META_COLS = [
    EXPERIMENT,
    FRAME,
    ACTUAL,
    BOUT_ID,
    BEHAVIOUR,
    ANIMAL,
    BATCH,
    CONDITION,
    CAMERA,
]


# -- loading --------------------------------------------------------------


def load_features_labels() -> pl.DataFrame:
    """Load aligned features and labels for every experiment that has features."""
    df = load_all_data(FEATURES_DIR, LABELS_DIR, BEHAVIOUR)
    return label_bouts(df, ACTUAL)


def feature_cols(df: pl.DataFrame) -> list[str]:
    """Return feature column names (everything that is not metadata)."""
    return [c for c in df.columns if c not in META_COLS]


def add_experiment_parts(df: pl.DataFrame) -> pl.DataFrame:
    """Add animal/batch/condition/camera columns parsed from EXPERIMENT."""
    return df.with_columns(
        pl.col(EXPERIMENT).str.split("-").list.first().alias(ANIMAL),
        pl.col(EXPERIMENT).str.split("-").list.get(1).alias(BATCH),
        pl.col(EXPERIMENT).str.split("-").list.get(2).alias(CONDITION),
        pl.col(EXPERIMENT).str.split("-").list.get(3).alias(CAMERA),
    )


def load_model_eval(model: str, split: str) -> pl.DataFrame:
    """Load a model's ``{split}_eval.parquet`` (prob/pred/actual per frame)."""
    return pl.read_parquet(MODELS_DIR / model / "evaluation" / f"{split}_eval.parquet")


# -- metrics --------------------------------------------------------------


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Average-precision (area under PR curve)."""
    return float(average_precision_score(y_true, y_prob))


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(y_true, y_prob))


def per_group_metrics(df: pl.DataFrame, group_cols: Sequence[str]) -> pl.DataFrame:
    """Per-group row count, positive count, PR-AUC and ROC-AUC.

    Assumes ``df`` carries ``ACTUAL`` and ``PROB`` columns.
    """
    rows: list[dict] = []
    for part in df.partition_by(list(group_cols), maintain_order=True):
        y = part[ACTUAL].to_numpy()
        p = part[PROB].to_numpy()
        n_pos = int(y.sum())
        row = {c: part[c][0] for c in group_cols}
        row["n"] = len(y)
        row["n_pos"] = n_pos
        row["pos_rate"] = n_pos / len(y)
        row["pr_auc"] = pr_auc(y, p) if n_pos > 0 else float("nan")
        row["roc_auc"] = roc_auc(y, p) if n_pos > 0 else float("nan")
        rows.append(row)
    return pl.DataFrame(rows)


def pooled_metrics(df: pl.DataFrame) -> dict[str, float]:
    """Pooled (all frames together) PR-AUC / ROC-AUC / positive rate."""
    y = df[ACTUAL].to_numpy()
    p = df[PROB].to_numpy()
    return {
        "n": len(y),
        "n_pos": int(y.sum()),
        "pos_rate": float(y.mean()),
        "pr_auc": pr_auc(y, p),
        "roc_auc": roc_auc(y, p),
    }


# -- model training helpers ------------------------------------------------


def split_by_test_experiments(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Split ``df`` into train/test using the xgb test eval's experiments."""
    test_exps = set(load_model_eval("xgb", "test")[EXPERIMENT].unique().to_list())
    feats = feature_cols(df)
    train = df.filter(~pl.col(EXPERIMENT).is_in(test_exps))
    test = df.filter(pl.col(EXPERIMENT).is_in(test_exps))
    return train, test, feats


def cap_rows(x: np.ndarray, y: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified random subsample to at most ``cap`` rows."""
    if len(y) <= cap:
        return x, y
    x_sub, _, y_sub, _ = train_test_split(
        x, y, train_size=cap, stratify=y, random_state=42
    )
    return x_sub, y_sub


def fit_xgb_eval(  # noqa: PLR0913
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_estimators: int = 150,
    max_depth: int = 5,
    scale_pos_weight: float = 10.0,
) -> dict[str, float]:
    """Fit a quick XGBoost and return pooled test PR-AUC / ROC-AUC."""
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    p = model.predict_proba(x_test)[:, 1]
    return {"pr_auc": pr_auc(y_test, p), "roc_auc": roc_auc(y_test, p)}


# -- reporting ------------------------------------------------------------


def write_json(obj: object, fp: Path) -> None:
    """Write a JSON-serialisable object to ``fp`` (creating parent dirs)."""
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(obj: object) -> object:
    """Coerce numpy scalars to native types for JSON serialisation."""
    if isinstance(obj, np.generic):
        return obj.item()
    msg = f"Object of type {type(obj)!r} is not JSON serialisable"
    raise TypeError(msg)
