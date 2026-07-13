"""Evaluation metrics and visualization for behavioural classifier."""

import json
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from behavysis.constants import ACTUAL, BOUT_ID, EXPERIMENT, FRAME, PROB

from .data import label_bouts

NIL = "nil"
BEHAV = "behav"
LABELS = [NIL, BEHAV]

SPLIT = "split"


def binary_report(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Summarise binary-classification performance for one split."""
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
) -> alt.Chart:
    """Build an overlaid line chart (one line per split) with a baseline."""
    return (
        alt.Chart(points)
        .mark_line()
        .encode(
            x=alt.X(f"{x_column}:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y(f"{y_column}:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(f"{SPLIT}:N", title="split"),
        )
        .properties(title=title, width=400, height=400)
    )


def _hist_chart(eval_df: pl.DataFrame, x_column: str, title: str) -> alt.Chart:
    return (
        alt.Chart(eval_df)
        .mark_bar(opacity=0.3, binSpacing=0)
        .encode(
            alt.X(f"{x_column}:Q", scale=alt.Scale(domain=[0, 1])).bin(maxbins=100),
            alt.Y(aggregate="count"),
            alt.Column(f"{ACTUAL}:N"),
            alt.Row(f"{SPLIT}:N"),
        )
        .properties(
            title=title, height=290, width=250, config={"axis": {"grid": False}}
        )
        .resolve_scale(y="independent")
    )


def save_eval_report(
    splits: dict[str, pl.DataFrame],
    eval_dir: Path,
    cv_summary: dict | None = None,
) -> dict[str, dict | pl.DataFrame | alt.Chart]:
    """Write ROC/PR charts and a JSON metric report for the given splits."""
    # Construct report
    report: dict[str, dict] = {
        name: binary_report(_df[ACTUAL].to_numpy(), _df[PROB].to_numpy())
        for name, (_df) in splits.items()
    }
    if cv_summary is not None:
        report["val"] = cv_summary

    # Construct data for plots
    eval_full_df = pl.concat(
        _df.with_columns(pl.lit(name).alias(SPLIT)) for name, _df in splits.items()
    )

    eval_bout_df = pl.concat(
        label_bouts(_df.sort([pl.col(EXPERIMENT), pl.col(FRAME)]))
        .group_by(BOUT_ID)
        .agg(
            pl.col(ACTUAL).max(),
            pl.col(PROB).max().alias("prob_max"),
            pl.col(PROB).mean().alias("prob_mean"),
        )
        .filter(pl.col(ACTUAL) == 1)
        .sort(BOUT_ID)
        .with_row_index("index")
        .with_columns(pl.lit(name).alias(SPLIT))
        for name, _df in splits.items()
    )
    roc_df = pl.concat(
        [
            _roc_points(_df[ACTUAL].to_numpy(), _df[PROB].to_numpy(), name)
            for name, _df in splits.items()
        ]
    )
    pr_df = pl.concat(
        [
            _pr_points(_df[ACTUAL].to_numpy(), _df[PROB].to_numpy(), name)
            for name, _df in splits.items()
        ]
    )
    # Make plots
    diagonal = (
        alt.Chart(pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))
        .mark_line(strokeDash=[4, 4], color="grey")
        .encode(x="x:Q", y="y:Q")
    )
    prob_hist_chart = _hist_chart(eval_full_df, PROB, "Prob histogram")
    bout_hist_chart = _hist_chart(
        eval_bout_df, "prob_max", "Max prob per-bout histogram"
    )
    roc_chart = _curve_chart(roc_df, "fpr", "tpr", "ROC curve") + diagonal
    pr_chart = _curve_chart(pr_df, "recall", "precision", "Precision-Recall curve")
    # Save
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "eval_report.json").write_text(json.dumps(report, indent=2))
    prob_hist_chart.save(eval_dir / "prob_hist.png")
    bout_hist_chart.save(eval_dir / "bout_hist.png")
    roc_chart.save(eval_dir / "roc.png")
    pr_chart.save(eval_dir / "pr.png")
    # Return
    return {
        # Report
        "report": report,
        # Data
        "eval_full_df": eval_full_df,
        "eval_bout_df": eval_bout_df,
        "roc_df": roc_df,
        "pr_df": pr_df,
        # Graphs
        "prob_hist_chart": prob_hist_chart,
        "bout_hist_chart": bout_hist_chart,
        "roc_chart": roc_chart,
        "pr_chart": pr_chart,
    }


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
    low_importance_threshold = 0.001
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
        n_low = int(np.sum(importances < low_importance_threshold))
        report["n_features_low_importance_lte_0.001"] = n_low

    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "feature_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Saved feature report to %s", eval_dir / "feature_report.json")
