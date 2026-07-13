"""Evaluation metrics and visualization for behavioural classifier."""

import json
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from loguru import logger
from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from behavysis.constants import ACTUAL, PROB

NIL = "nil"
BEHAV = "behav"
LABELS = [NIL, BEHAV]

SPLIT = "split"


def binary_report(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Summarise binary-classification performance for one split.

    Reports threshold-independent metrics (ROC AUC, Gini, PR AUC) plus the
    Youden-optimal threshold and the classification metrics achieved at it.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (1 = behaviour).
    y_prob : np.ndarray
        Predicted probabilities for the positive class.

    Returns:
    -------
    dict
        Metric summary, or an empty dict if only one class is present
        (ROC/PR AUC are undefined in that case).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if len(np.unique(y_true)) < 2:  # noqa: PLR2004
        logger.warning("Only one class present; ROC/PR metrics undefined.")
        return {}

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    best = int(np.argmax(youden))
    ideal_threshold = float(thresholds[best])

    y_pred = (y_prob >= ideal_threshold).astype(int)
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    return {
        "n": len(y_true),
        "n_positive": int(np.sum(y_true == 1)),
        "roc_auc": roc_auc,
        "gini": 2.0 * roc_auc - 1.0,
        "pr_auc": pr_auc,
        "ideal_threshold": ideal_threshold,
        "precision": report[BEHAV]["precision"],
        "recall": report[BEHAV]["recall"],
        "f1": report[BEHAV]["f1-score"],
        "accuracy": report["accuracy"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _roc_points(y_true: np.ndarray, y_prob: np.ndarray, split: str) -> pl.DataFrame:
    """ROC curve points (fpr, tpr) as a long-form DataFrame."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return pl.DataFrame({"fpr": fpr, "tpr": tpr, SPLIT: split})


def _pr_points(y_true: np.ndarray, y_prob: np.ndarray, split: str) -> pl.DataFrame:
    """Precision-recall curve points (recall, precision) as long-form."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return pl.DataFrame({"recall": recall, "precision": precision, SPLIT: split})


def _curve_chart(
    points: pl.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    baseline: alt.Chart | None,
) -> alt.Chart:
    """Build an overlaid line chart (one line per split) with a baseline."""
    line = (
        alt.Chart(points)
        .mark_line()
        .encode(
            x=alt.X(f"{x_column}:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y(f"{y_column}:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(f"{SPLIT}:N", title="split"),
        )
        .properties(title=title, width=400, height=400)
    )
    return line + baseline if baseline is not None else line


def save_eval_report(
    splits: dict[str, pl.DataFrame],
    eval_dir: Path,
    cv_summary: dict | None = None,
) -> None:
    """Write ROC/PR charts and a JSON metric report for the given splits."""
    eval_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict] = {
        name: binary_report(eval_df[ACTUAL].to_numpy(), eval_df[PROB].to_numpy())
        for name, (eval_df) in splits.items()
    }
    if cv_summary is not None:
        report["val"] = cv_summary
    (eval_dir / "eval_report.json").write_text(json.dumps(report, indent=2))

    roc_pts = pl.concat(
        [
            _roc_points(eval_df[ACTUAL].to_numpy(), eval_df[PROB].to_numpy(), name)
            for name, eval_df in splits.items()
        ]
    )
    pr_pts = pl.concat(
        [
            _pr_points(eval_df[ACTUAL].to_numpy(), eval_df[PROB].to_numpy(), name)
            for name, eval_df in splits.items()
        ]
    )

    diagonal = (
        alt.Chart(pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))
        .mark_line(strokeDash=[4, 4], color="grey")
        .encode(x="x:Q", y="y:Q")
    )
    roc_chart = _curve_chart(roc_pts, "fpr", "tpr", "ROC curve", diagonal)
    pr_chart = _curve_chart(
        pr_pts, "recall", "precision", "Precision-Recall curve", None
    )

    roc_chart.save(eval_dir / "roc.png")
    pr_chart.save(eval_dir / "pr.png")
    logger.info("Saved eval report and ROC/PR charts to {}", eval_dir)


def eval_metrics_pcutoffs(y_true: np.ndarray, y_prob: np.ndarray) -> Figure:
    """Plot precision, recall, f1, and accuracy across probability cutoffs."""
    labels = LABELS
    pcutoffs = np.linspace(0, 1, 101)
    precisions = np.zeros(pcutoffs.shape[0])
    recalls = np.zeros(pcutoffs.shape[0])
    f1 = np.zeros(pcutoffs.shape[0])
    accuracies = np.zeros(pcutoffs.shape[0])

    for i, pcutoff in enumerate(pcutoffs):
        y_pred = y_prob > pcutoff
        report = classification_report(
            y_true,
            y_pred,
            target_names=labels,
            output_dict=True,
        )
        precisions[i] = report[BEHAV]["precision"]
        recalls[i] = report[BEHAV]["recall"]
        f1[i] = report[BEHAV]["f1-score"]
        accuracies[i] = report["accuracy"]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.lineplot(x=pcutoffs, y=precisions, label="precision", ax=ax)
    sns.lineplot(x=pcutoffs, y=recalls, label="recall", ax=ax)
    sns.lineplot(x=pcutoffs, y=f1, label="f1", ax=ax)
    sns.lineplot(x=pcutoffs, y=accuracies, label="accuracy", ax=ax)
    return fig


def eval_logc(y_true: np.ndarray, y_prob: np.ndarray, pcutoff: float) -> Figure:
    """Plot logistic curve of predicted probabilities vs true labels."""
    rng = np.random.default_rng()
    y_eval = pd.DataFrame(
        {
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": y_prob > pcutoff,
            "y_true_jitter": y_true + (0.2 * (rng.random(len(y_prob)) - 0.5)),
        },
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=y_eval,
        x="y_prob",
        y="y_true_jitter",
        marker=".",
        s=10,
        linewidth=0,
        alpha=0.2,
        ax=ax,
    )
    pcutoffs = np.linspace(0, 1, 101)
    ratios = np.vectorize(lambda i: np.mean(i > y_eval["y_prob"]))(pcutoffs)
    sns.lineplot(x=pcutoffs, y=ratios, ax=ax)
    return fig


def save_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    eval_dir: Path,
    *,
    top_n: int = 30,
) -> None:
    """Save feature importance bar chart."""
    n = min(top_n, len(importances))
    if n == 0:
        return
    idx = np.argsort(importances)[-n:]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.3)))
    ax.barh(range(n), vals, color="steelblue")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {n} Features by Importance")
    fig.tight_layout()
    fig.savefig(eval_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
    logger.info(
        "Saved feature importance plot to %s",
        eval_dir / "feature_importance.png",
    )


def save_feature_report(
    feature_names: list[str],
    importances: np.ndarray | None,
    eval_dir: Path,
    n_features_total: int | None = None,
) -> None:
    """Save feature count and importance summary as JSON."""
    n_used = len(feature_names)
    report: dict = {
        "n_features_total": (
            n_features_total if n_features_total is not None else n_used
        ),
        "n_features_used": n_used,
    }

    if importances is not None:
        non_zero = int(np.sum(importances > 0))
        report["n_features_non_zero_importance"] = non_zero
        n_low = int(np.sum(importances < 0.001))
        report["n_features_low_importance_lte_0.001"] = n_low

    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "feature_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Saved feature report to %s", eval_dir / "feature_report.json")
