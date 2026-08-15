"""CV grouping leak investigation.

Tests whether hyperparameter-selection cross-validation that groups by
``BOUT_ID`` (as ``SklearnAdapter.fit`` does) leaks information relative to
grouping by ``EXPERIMENT``.

For each grouping we:

- train XGBoost (fixed hyperparameters) on 3-fold StratifiedGroupKFold and
  report the mean held-out PR-AUC, and
- count how many *experiments* leak between train and validation folds
  (a bout-grouped split can place different bouts of the same experiment on
  both sides; an experiment-grouped split cannot).

Output is written as JSON to ``data/front-rear/eda/``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold

from behavysis.constants import BOUT_ID, EXPERIMENT

from .common import (
    ACTUAL,
    EDA_OUT_DIR,
    cap_rows,
    feature_cols,
    fit_xgb_eval,
    load_features_labels,
    split_by_test_experiments,
    write_json,
)

_ROW_CAP = 250_000
_N_ESTIMATORS = 120
_N_SPLITS = 3


def _cv_by_grouping(df: pl.DataFrame, group_col: str) -> dict:
    """Mean held-out PR-AUC and experiment-leak count for one grouping."""
    feats = feature_cols(df)
    x = df.select(feats).cast(pl.Float32).to_numpy()
    y = df[ACTUAL].to_numpy()
    groups = df[group_col].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=_N_SPLITS, shuffle=True, random_state=42)

    scores: list[float] = []
    n_leaked_experiments = 0
    n_folds = 0
    for tr_idx, te_idx in sgkf.split(x, y, groups):
        tr_exps = set(df[EXPERIMENT][tr_idx].to_list())
        te_exps = set(df[EXPERIMENT][te_idx].to_list())
        n_leaked_experiments += len(tr_exps & te_exps)
        n_folds += 1
        x_tr, y_tr = cap_rows(x[tr_idx], y[tr_idx], _ROW_CAP)
        scores.append(
            fit_xgb_eval(x_tr, y_tr, x[te_idx], y[te_idx], n_estimators=_N_ESTIMATORS)[
                "pr_auc"
            ]
        )
    return {
        "mean_pr_auc": float(np.mean(scores)),
        "scores": [float(s) for s in scores],
        "n_experiments": df[EXPERIMENT].n_unique(),
        "n_experiments_leaked_per_fold": n_leaked_experiments / n_folds,
    }


def main() -> None:
    """Run the grouping comparison on the training split."""
    df, _, _ = split_by_test_experiments(load_features_labels())
    report = {
        "by_bout_id": _cv_by_grouping(df, BOUT_ID),
        "by_experiment": _cv_by_grouping(df, EXPERIMENT),
    }
    write_json(report, EDA_OUT_DIR / "cv_leak.json")
    for name, res in report.items():
        print(  # noqa: T201
            f"{name}: mean_pr_auc={res['mean_pr_auc']:.3f} "
            f"scores={[f'{s:.3f}' for s in res['scores']]} "
            f"experiments_leaked_per_fold={res['n_experiments_leaked_per_fold']:.1f}"
        )


if __name__ == "__main__":
    main()
