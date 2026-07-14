"""Evaluation metrics and visualization for behavioural classifier."""

import json
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from loguru import logger
from pydantic import BaseModel
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from behavysis.behaviour_classifier.data import agg_eval_df_by_bouts
from behavysis.constants import ACTUAL, PRED, PROB

NIL = "nil"
BEHAV = "behav"
LABELS = [NIL, BEHAV]

SPLIT = "split"

# ----- Typing ------------------------------------------------------


class EvalReport(BaseModel):
    """Eval report."""

    report: dict[str, object]
    df: dict[str, pl.DataFrame]
    chart: dict[str, alt.Chart]


# ----- Functions ---------------------------------------------------


def binary_report(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    """Summarise binary-classification performance for one split."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if len(np.unique(y_true)) < 2:  # noqa: PLR2004
        logger.warning("Only one class present; ROC/PR metrics undefined.")
        return {}

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

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
        "precision": report[BEHAV]["precision"],
        "recall": report[BEHAV]["recall"],
        "f1": report[BEHAV]["f1-score"],
        "accuracy": report["accuracy"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _roc_df(y_true: np.ndarray, y_prob: np.ndarray) -> pl.DataFrame:
    """ROC curve points (fpr, tpr) as a long-form DataFrame."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return pl.DataFrame({"fpr": fpr, "tpr": tpr}).sort("fpr")


def _pr_df(y_true: np.ndarray, y_prob: np.ndarray) -> pl.DataFrame:
    """Precision-recall curve points (recall, precision) as long-form."""
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_prob, drop_intermediate=True
    )
    return (
        pl.DataFrame(
            {
                "recall": recall[:-1],
                "precision": precision[:-1],
                "thresholds": thresholds,
            }
        )
        .group_by("recall")
        .mean()
        .sort("recall")
    )


def _curve_chart(
    df: pl.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    layer: alt.Chart | None = None,
) -> alt.Chart:
    """Build an overlaid line chart (one line per split) with a baseline."""
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(f"{x_column}:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y(f"{y_column}:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(f"{SPLIT}:N", title="split"),
        )
    )
    if layer:
        chart = chart + layer
    return chart.properties(
        title=title, width=400, height=400, config={"axis": {"grid": True}}
    )


def _hist_chart(df: pl.DataFrame, x_column: str, title: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar(opacity=0.3, binSpacing=0)
        .encode(
            alt.X(f"{x_column}:Q", scale=alt.Scale(domain=[0, 1])).bin(maxbins=50),
            alt.Y(aggregate="count"),
            alt.Column(f"{ACTUAL}:N"),
            alt.Row(f"{SPLIT}:N"),
        )
        .properties(
            title=title, height=290, width=250, config={"axis": {"grid": False}}
        )
        .resolve_scale(y="independent")
    )


def make_eval_report(splits: dict[str, pl.DataFrame]) -> EvalReport:
    """Make ROC/PR charts and a JSON metric report for the given splits."""
    # Construct bouts splits eval (bouts equivalent of splits)
    bouts_splits = {_name: agg_eval_df_by_bouts(_df) for _name, _df in splits.items()}
    # Prepare to store eval results
    res_report: dict[str, object] = {}
    res_df: dict[str, pl.DataFrame] = {}
    res_chart: dict[str, alt.Chart] = {}
    # For both frames and bouts evals
    for _splits_name, _splits_data in {"frames": splits, "bouts": bouts_splits}.items():
        # Make report
        res_report[f"{_splits_name}_report"] = {
            _name: binary_report(
                _df[ACTUAL].to_numpy(), _df[PROB].to_numpy(), _df[PRED].to_numpy()
            )
            for _name, (_df) in _splits_data.items()
        }
        # Make eval dataframes
        res_df[f"{_splits_name}_eval_df"] = pl.concat(
            _df.with_columns(pl.lit(_name).alias(SPLIT))
            for _name, _df in _splits_data.items()
        )
        res_df[f"{_splits_name}_roc_df"] = pl.concat(
            [
                _roc_df(_df[ACTUAL].to_numpy(), _df[PROB].to_numpy()).with_columns(
                    pl.lit(_name).alias(SPLIT)
                )
                for _name, _df in _splits_data.items()
            ]
        )
        res_df[f"{_splits_name}_pr_df"] = pl.concat(
            [
                _pr_df(_df[ACTUAL].to_numpy(), _df[PROB].to_numpy()).with_columns(
                    pl.lit(_name).alias(SPLIT)
                )
                for _name, _df in _splits_data.items()
            ]
        )
        # Make plots
        diagonal = (
            alt.Chart(pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))
            .mark_line(strokeDash=[4, 4], color="grey")
            .encode(x="x:Q", y="y:Q")
        )
        res_chart[f"{_splits_name}_hist_chart"] = _hist_chart(
            res_df[f"{_splits_name}_eval_df"], PROB, f"Prob Histogram {_splits_name}"
        )
        res_chart[f"{_splits_name}_roc_chart"] = _curve_chart(
            res_df[f"{_splits_name}_roc_df"], "fpr", "tpr", "ROC curve", diagonal
        )
        res_chart[f"{_splits_name}_pr_chart"] = _curve_chart(
            res_df[f"{_splits_name}_pr_df"], "recall", "precision", "PR curve"
        )
        res_chart[f"{_splits_name}_thresholds_chart"] = (
            alt.Chart(
                res_df[f"{_splits_name}_pr_df"].unpivot(index=[SPLIT, "thresholds"])
            )
            .mark_line()
            .encode(
                x=alt.X("thresholds:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(field="variable", type="nominal"),
                row=alt.Row(field=SPLIT, type="nominal"),
            )
            .properties(
                title="PR at each Threshold",
                width=400,
                height=400,
                config={"axis": {"grid": True}},
            )
        )
    # Return
    return EvalReport(report=res_report, df=res_df, chart=res_chart)


def save_eval_report(splits: dict[str, pl.DataFrame], eval_dir: Path) -> None:
    """Write ROC/PR charts and a JSON metric report for the given splits."""
    # Make eval dir
    res = make_eval_report(splits)
    # Save. Only report and charts, not df
    eval_dir.mkdir(parents=True, exist_ok=True)
    for _name, _report in res.report.items():
        (eval_dir / f"{_name}.json").write_text(yaml.dump(_report))
    for _name, _chart in res.chart.items():
        _chart.save(eval_dir / f"{_name}.png")


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
