"""Evaluation metrics and visualization for behavioural classifier."""

from typing import TypedDict

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from behavysis.behaviour_classifier.data import (
    agg_eval_df_by_bouts,
    df_get_features,
    df_get_labels,
)
from behavysis.constants import ACTUAL, BOUT, FRAME, PRED, PROB

NIL = "nil"
BEHAV = "behav"
LABELS = [NIL, BEHAV]

SPLIT = "split"


# ----- schemas -----------------------------------------------------


class EvalResult(TypedDict):
    """Eval result."""

    report: dict[str, dict[str, dict[str, float]]]
    df: dict[str, pl.DataFrame]
    chart: dict[str, alt.Chart]


# ----- Helper Functions ---------------------------------------------------


def _precision_at_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float,
) -> float:
    """Best achievable precision while maintaining recall >= target.

    Computed from the full precision-recall curve. Answers: "if I want
    to catch at least X% of behaviour, what precision can I expect?"
    This is the single most decision-relevant metric for a screening
    classifier — higher recall costs precision (more false positives to
    review), and this function quantifies that trade-off at the target
    operating point.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities.
    target_recall : float
        Minimum acceptable recall (e.g. 0.99).

    Returns:
    --------
    float
        Maximum precision at any threshold where recall >= target.
        0.0 if the target recall cannot be achieved.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    mask = recall >= target_recall
    if mask.any():
        return float(np.max(precision[mask]))
    return 0.0


def _precision_at_target_recalls(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute precision at each target recall checkpoint."""
    return {
        label: _precision_at_recall(y_true, y_prob, threshold)
        for threshold, label in (
            (0.99, "precision_at_recall_099"),
            (1.00, "precision_at_recall_100"),
        )
    }


def _report(
    y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
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
        **_precision_at_target_recalls(y_true, y_prob),
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


# ----- Bout health metrics -------------------------------------------


def _bout_health(bout_df: pl.DataFrame) -> dict[str, float]:
    """Compute temporal-quality metrics from per-bout eval data.

    Goes beyond binary bout detection (did we overlap at all?) to
    answer questions about *how well* the model covers behavioral
    episodes in time.

    Metrics computed
    ----------------
    bout_detection_rate_50pct : float
        Fraction of real behavioural bouts (``ACTUAL.max() == 1``)
        where the model predicts positive on >= 50 % of frames
        (``pred_mean >= 0.5``).  A stricter alternative to the raw
        bout recall which only requires a single overlapping frame.
    bout_detection_rate_any : float
        Fraction of real bouts with at least one predicted positive
        frame (``PRED.max() == 1``).  Identical to bout-level recall
        from the standard report; included here for convenience.
    mean_coverage : float
        Mean ``pred_mean`` across real bouts.  If this is 0.85, the
        typical real behavioural episode has 85 % of its frames caught
        by the classifier.
    fragmentation : float
        ``n_predicted_bouts / n_real_bouts``.  Values > 1 indicate
        that the model is splitting real episodes into multiple
        predicted fragments.  High fragmentation may cause a human
        reviewer to mark the same real event multiple times.
    mean_bout_len_real : float
        Average duration (in frames) of true behavioural bouts.
    """
    real_bouts = bout_df.filter(pl.col(ACTUAL) == 1)
    pred_bouts = bout_df.filter(pl.col(PRED) == 1)

    n_real = real_bouts.height
    n_pred = pred_bouts.height

    return {
        "bout_detection_rate_any": float(
            real_bouts.select(pl.col(PRED).max().mean()).item()
        )
        if n_real > 0
        else 0.0,
        "bout_detection_rate_50pct": float(
            real_bouts.select((pl.col(f"{PRED}_mean") >= 0.5).mean()).item()  # noqa: PLR2004
        )
        if n_real > 0
        else 0.0,
        "mean_coverage": float(real_bouts.select(pl.col(f"{PRED}_mean").mean()).item())
        if n_real > 0
        else 0.0,
        "fragmentation": n_pred / n_real if n_real > 0 else 0.0,
        "mean_bout_len_real": float(
            real_bouts.select(pl.col("bout_n_frames").mean()).item()
        )
        if n_real > 0
        else 0.0,
    }


# ----- API Functions ---------------------------------------------------


def make_eval_result(splits: dict[str, pl.DataFrame]) -> EvalResult:
    """Make ROC/PR charts and a YAML metric report for the given splits."""
    # Construct bouts splits eval (bouts equivalent of splits)
    bouts_splits = {_name: agg_eval_df_by_bouts(_df) for _name, _df in splits.items()}
    # Prepare to store eval results
    res_report: dict[str, dict[str, dict[str, float]]] = {}
    res_df: dict[str, pl.DataFrame] = {}
    res_chart: dict[str, alt.Chart] = {}
    # For both frames and bouts evals
    for _splits_name, _splits_data in {FRAME: splits, BOUT: bouts_splits}.items():
        # Make report
        res_report[f"{_splits_name}_report"] = {
            _name: _report(
                _df[ACTUAL].to_numpy(), _df[PROB].to_numpy(), _df[PRED].to_numpy()
            )
            for _name, _df in _splits_data.items()
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
    # Bout health
    res_report[f"{BOUT}_health"] = {
        _name: _bout_health(_df) for _name, _df in bouts_splits.items()
    }
    # Review efficiency (frame-level: predicted-positive / true-positive frames)
    res_report["review_efficiency"] = {
        _name: {
            "efficiency": float(
                _df.select(pl.col(PRED).sum() / pl.col(ACTUAL).sum()).item()
            )
        }
        if _df.select(pl.col(ACTUAL).sum()).item() > 0
        else {"efficiency": 0.0}
        for _name, _df in splits.items()
    }
    # Return
    return EvalResult(report=res_report, df=res_df, chart=res_chart)


# ----- Explainability -------------------------------------------


def compute_shap(
    model: Pipeline,
    df: pl.DataFrame,
    *,
    top_n: int = 30,
    max_samples: int = 500,
) -> dict:
    """Compute shap for tree-based model."""
    preprocessor = model[:-1]
    clf = model.steps[-1][1]

    features_df = df_get_features(df)
    features_df = preprocessor.transform(features_df)
    feature_names = preprocessor.get_feature_names_out()
    y_df = df_get_labels(df)

    if len(features_df) > max_samples:
        idx, _ = train_test_split(
            np.arange(features_df.shape[0]),
            train_size=max_samples,
            stratify=y_df,
            random_state=42,
        )
        features_df = features_df[idx]

    shap_explainer = shap.TreeExplainer(clf)
    shap_values = shap_explainer.shap_values(features_df)
    shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pl.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort("mean_abs_shap", descending=True)

    top_idx = np.argsort(mean_abs_shap)[-top_n:]
    shap.summary_plot(
        shap_values[:, top_idx],
        features_df[:, top_idx],
        feature_names=feature_names[top_idx],
        show=False,
    )

    return {
        "importance_df": importance_df,
        "fig": plt.gcf(),
    }
