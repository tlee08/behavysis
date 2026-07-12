"""Evaluation metrics and visualization for behavioural classifier."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.figure import Figure
from sklearn.metrics import classification_report, confusion_matrix

from behavysis.constants import ACTUAL, PRED, PROB

NIL = "nil"
BEHAV = "behav"
LABELS = [NIL, BEHAV]


def eval_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Generate classification report with precision, recall, f1-score."""
    labels = LABELS
    return classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=labels,
        output_dict=True,
    )


def eval_conf_matr(y_true: np.ndarray, y_pred: np.ndarray) -> Figure:
    """Generate confusion matrix heatmap."""
    labels = LABELS
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(
        confusion_matrix(y_true, y_pred),
        annot=True,
        fmt="d",
        cmap="viridis",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return fig


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


def save_training_history(history: pd.DataFrame, eval_dir: Path) -> None:
    """Save training history dataframe and plot.

    Parameters
    ----------
    history : pd.DataFrame
        Training history with loss values.
    eval_dir : Path
        Directory for evaluation outputs.
    """
    history.to_parquet(eval_dir / "history.parquet")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.lineplot(data=history, ax=ax)
    fig.savefig(eval_dir / "history.png")
    plt.close(fig)


def save_evaluation_results(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    behaviour_name: str,
    pcutoff: float,
    eval_dir: Path,
    name: str,
    index_ls: list[np.ndarray],
) -> tuple[pd.DataFrame, dict, Figure, Figure, Figure]:
    """Generate and save evaluation results."""
    # Build evaluation dataframe
    index = pd.Index(np.arange(np.concatenate(index_ls).shape[0]), name="frame")
    columns = pd.MultiIndex.from_tuples(
        [(behaviour_name, PROB), (behaviour_name, PRED), (behaviour_name, ACTUAL)],
        names=["behaviour", "outcomes"],
    )
    eval_df = pd.DataFrame(
        np.column_stack([y_prob, y_pred, y_true]),
        index=index,
        columns=columns,
    )

    # Generate reports
    report_dict = eval_report(y_true, y_pred)
    conf_matr_fig = eval_conf_matr(y_true, y_pred)
    pcutoffs_fig = eval_metrics_pcutoffs(y_true, y_prob)
    logc_fig = eval_logc(y_true, y_prob, pcutoff)

    # Save outputs
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_df.to_parquet(eval_dir / f"{name}_eval.parquet")
    (eval_dir / f"{name}_report.json").write_text(json.dumps(report_dict, indent=2))
    conf_matr_fig.savefig(eval_dir / f"{name}_confm.png")
    pcutoffs_fig.savefig(eval_dir / f"{name}_pcutoffs.png")
    logc_fig.savefig(eval_dir / f"{name}_logc.png")

    plt.close(conf_matr_fig)
    plt.close(pcutoffs_fig)
    plt.close(logc_fig)

    return eval_df, report_dict, conf_matr_fig, pcutoffs_fig, logc_fig


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
