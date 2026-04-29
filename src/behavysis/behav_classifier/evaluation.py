"""Evaluation metrics and visualization for behavioral classifier."""

import json
import logging
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import classification_report, confusion_matrix

from behavysis.df_classes.behav_classifier_df import BehavClassifierEvalDf
from behavysis.df_classes.behav_df import BehavScoredDf
from behavysis.utils.df_mixin import DFMixin
from behavysis.utils.misc_utils import enum2tuple

logger = logging.getLogger(__name__)


class GenericBehavLabels(Enum):
    """Standard behavior label names for classification reports."""

    NIL = "nil"
    BEHAV = "behav"


def eval_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Generate classification report with precision, recall, f1-score.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    dict
        Classification report dictionary.
    """
    return classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=enum2tuple(GenericBehavLabels),
        output_dict=True,
    )


def eval_conf_matr(y_true: np.ndarray, y_pred: np.ndarray) -> Figure:
    """Generate confusion matrix heatmap.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    Figure
        Matplotlib figure with confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(
        confusion_matrix(y_true, y_pred),
        annot=True,
        fmt="d",
        cmap="viridis",
        cbar=False,
        xticklabels=enum2tuple(GenericBehavLabels),
        yticklabels=enum2tuple(GenericBehavLabels),
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return fig


def eval_metrics_pcutoffs(y_true: np.ndarray, y_prob: np.ndarray) -> Figure:
    """Plot precision, recall, f1, and accuracy across probability cutoffs.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_prob : np.ndarray
        Predicted probabilities.

    Returns
    -------
    Figure
        Matplotlib figure with metric curves.
    """
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
            target_names=enum2tuple(GenericBehavLabels),
            output_dict=True,
        )
        precisions[i] = report[GenericBehavLabels.BEHAV.value]["precision"]
        recalls[i] = report[GenericBehavLabels.BEHAV.value]["recall"]
        f1[i] = report[GenericBehavLabels.BEHAV.value]["f1-score"]
        accuracies[i] = report["accuracy"]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.lineplot(x=pcutoffs, y=precisions, label="precision", ax=ax)
    sns.lineplot(x=pcutoffs, y=recalls, label="recall", ax=ax)
    sns.lineplot(x=pcutoffs, y=f1, label="f1", ax=ax)
    sns.lineplot(x=pcutoffs, y=accuracies, label="accuracy", ax=ax)
    return fig


def eval_logc(y_true: np.ndarray, y_prob: np.ndarray) -> Figure:
    """Plot logistic curve of predicted probabilities vs true labels.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_prob : np.ndarray
        Predicted probabilities.

    Returns
    -------
    Figure
        Matplotlib figure with logistic curve.
    """
    y_eval = pd.DataFrame(
        {
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": y_prob > 0.4,
            "y_true_jitter": y_true + (0.2 * (np.random.rand(len(y_prob)) - 0.5)),
        }
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


def eval_bouts(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Analyze prediction accuracy by behavioral bout.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    pd.DataFrame
        Summary of prediction accuracy per bout.
    """
    y_eval = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    y_eval["ids"] = np.cumsum(y_eval["y_true"] != y_eval["y_true"].shift())
    y_eval_grouped = y_eval.groupby("ids")
    y_eval_summary = pd.DataFrame(
        y_eval_grouped.apply(lambda x: (x["y_pred"] == x["y_true"]).mean()),
        columns=["proportion"],
    )
    y_eval_summary["actual_bout"] = y_eval_grouped.apply(lambda x: x["y_true"].mean())
    y_eval_summary["bout_len"] = y_eval_grouped.apply(lambda x: x.shape[0])
    y_eval_summary = y_eval_summary.sort_values("proportion")
    return y_eval_summary


def save_training_history(history: pd.DataFrame, eval_dir: Path) -> None:
    """Save training history dataframe and plot.

    Parameters
    ----------
    history : pd.DataFrame
        Training history with loss values.
    eval_dir : Path
        Directory for evaluation outputs.
    """
    DFMixin.write(history, eval_dir / f"history.{DFMixin.IO}")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.lineplot(data=history, ax=ax)
    fig.savefig(eval_dir / "history.png")
    plt.close(fig)


def save_evaluation_results(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    behav_name: str,
    pcutoff: float,
    eval_dir: Path,
    name: str,
    index_ls: list[np.ndarray],
) -> tuple[pd.DataFrame, dict, Figure, Figure, Figure]:
    """Generate and save evaluation results.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_prob : np.ndarray
        Predicted probabilities.
    y_pred : np.ndarray
        Predicted labels.
    behav_name : str
        Name of the behavior.
    pcutoff : float
        Probability cutoff used.
    eval_dir : Path
        Directory for evaluation outputs.
    name : str
        Name for output files (e.g., "train", "test").
    index_ls : list[np.ndarray]
        List of index arrays for each dataframe.

    Returns
    -------
    tuple
        (eval_df, report_dict, conf_matr_fig, pcutoffs_fig, logc_fig)
    """
    # Build evaluation dataframe
    eval_df = BehavClassifierEvalDf.init_df(
        pd.Series(np.arange(np.concatenate(index_ls).shape[0]))
    )
    eval_df[(behav_name, BehavClassifierEvalDf.OutcomesCols.PROB.value)] = y_prob
    eval_df[(behav_name, BehavClassifierEvalDf.OutcomesCols.PRED.value)] = y_pred
    eval_df[(behav_name, BehavScoredDf.OutcomesCols.ACTUAL.value)] = y_true

    # Generate reports
    report_dict = eval_report(y_true, y_pred)
    conf_matr_fig = eval_conf_matr(y_true, y_pred)
    pcutoffs_fig = eval_metrics_pcutoffs(y_true, y_prob)
    logc_fig = eval_logc(y_true, y_prob)

    # Save outputs
    BehavClassifierEvalDf.write(eval_df, eval_dir / f"{name}_eval.{BehavClassifierEvalDf.IO}")
    (eval_dir / f"{name}_report.json").write_text(json.dumps(report_dict, indent=2))
    conf_matr_fig.savefig(eval_dir / f"{name}_confm.png")
    pcutoffs_fig.savefig(eval_dir / f"{name}_pcutoffs.png")
    logc_fig.savefig(eval_dir / f"{name}_logc.png")

    plt.close(conf_matr_fig)
    plt.close(pcutoffs_fig)
    plt.close(logc_fig)

    return eval_df, report_dict, conf_matr_fig, pcutoffs_fig, logc_fig
