"""Adversarial / shortcut validation.

Trains lightweight classifiers to answer: "how much does the feature vector
itself leak group identity?"  If features can cleanly separate train from
test, or HOT from COLD, or identify the animal, then a behavioural
classifier can exploit those shortcuts instead of learning the behaviour.

Checks (all on the raw per-frame features):

- train-vs-test experiment identity  -> ROC-AUC (near 1 means domain shift)
- HOT vs COLD condition              -> ROC-AUC
- animal identity (32 animals)       -> top-1 accuracy

Output is written as JSON to ``data/front-rear/eda/``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from behavysis.constants import EXPERIMENT

from .common import (
    ANIMAL,
    CONDITION,
    EDA_OUT_DIR,
    add_experiment_parts,
    feature_cols,
    load_features_labels,
    load_model_eval,
    write_json,
)

if TYPE_CHECKING:
    import numpy as np

_MAX_BINARY_SAMPLES = 100_000
_MAX_MULTI_SAMPLES = 40_000
_N_FOLDS = 5


def _subsample(x: np.ndarray, y: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified random subsample to at most ``cap`` rows."""
    if len(y) <= cap:
        return x, y
    x_sub, _, y_sub, _ = train_test_split(
        x, y, train_size=cap, stratify=y, random_state=42
    )
    return x_sub, y_sub


def _binary_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Cross-validated ROC-AUC of a logreg on standardised features."""
    x, y = _subsample(x, y, _MAX_BINARY_SAMPLES)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    scores = cross_val_score(
        model,
        x,
        y,
        cv=StratifiedKFold(_N_FOLDS, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=-1,
    )
    return float(scores.mean())


def _animal_accuracy(x: np.ndarray, y: np.ndarray) -> float:
    """Cross-validated top-1 accuracy for animal identity."""
    x, y = _subsample(x, y, _MAX_MULTI_SAMPLES)
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    scores = cross_val_score(
        model,
        x,
        y,
        cv=StratifiedKFold(_N_FOLDS, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
    )
    return float(scores.mean())


def _test_experiments() -> set[str]:
    """Return the held-out test experiment names (from the xgb test eval)."""
    return set(load_model_eval("xgb", "test")[EXPERIMENT].unique().to_list())


def main() -> None:
    """Run adversarial checks and write the report."""
    df = add_experiment_parts(load_features_labels())
    feats = feature_cols(df)
    x = df.select(feats).cast(pl.Float32).to_numpy()

    test_exps = _test_experiments()
    is_test = df[EXPERIMENT].is_in(test_exps).to_numpy().astype(int)

    report = {
        "n_features": len(feats),
        "train_test_auc": _binary_auc(x, is_test),
        "condition_auc": _binary_auc(
            x, (df[CONDITION] == "HOT").to_numpy().astype(int)
        ),
        "animal_accuracy": _animal_accuracy(x, df[ANIMAL].to_numpy()),
        "animal_chance": 1.0 / df[ANIMAL].n_unique(),
    }
    write_json(report, EDA_OUT_DIR / "adversarial.json")
    print(  # noqa: T201
        f"train_test_auc={report['train_test_auc']:.3f} "
        f"condition_auc={report['condition_auc']:.3f} "
        f"animal_acc={report['animal_accuracy']:.3f} "
        f"(chance={report['animal_chance']:.3f})"
    )


if __name__ == "__main__":
    main()
