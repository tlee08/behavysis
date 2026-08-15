"""Improvement experiments.

Each experiment isolates one suspected bottleneck from the diagnosis:

1. ``grouped_cv``     — honest grouped cross-validation, by EXPERIMENT and by
                        ANIMAL.  The gap between the two shows how much the
                        "same animal in train and test" protocol inflates the
                        apparent ceiling.
2. ``cross_condition`` — train on HOT and test on COLD (and vice versa) to
                        quantify the condition-level distribution shift.
3. ``temporal_features`` — baseline (200 features) vs. added lag-1/lag-2 of
                        the 8 primitive features, testing whether temporal
                        context is under-represented.

Output is written as JSON to ``data/front-rear/eda/``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from behavysis.constants import EXPERIMENT

from .common import (
    ACTUAL,
    ANIMAL,
    CONDITION,
    EDA_OUT_DIR,
    add_experiment_parts,
    cap_rows,
    feature_cols,
    fit_xgb_eval,
    load_features_labels,
    split_by_test_experiments,
    write_json,
)

_ROW_CAP = 250_000
_N_ESTIMATORS = 120
_CV_SPLITS = 5


def _primitive_cols(feats: list[str]) -> list[str]:
    """The 8 primitive features (no rolling ``_w`` suffix)."""
    return [f for f in feats if "_w" not in f]


def _grouped_cv(df: pl.DataFrame, group_col: str) -> float:
    """Mean test PR-AUC from GroupKFold over ``group_col``."""
    feats = feature_cols(df)
    x = df.select(feats).cast(pl.Float32).to_numpy()
    y = df[ACTUAL].to_numpy()
    groups = df[group_col].to_numpy()
    gkf = GroupKFold(n_splits=_CV_SPLITS)
    scores = []
    for tr_idx, te_idx in gkf.split(x, y, groups):
        x_tr, y_tr = cap_rows(x[tr_idx], y[tr_idx], _ROW_CAP)
        scores.append(
            fit_xgb_eval(x_tr, y_tr, x[te_idx], y[te_idx], n_estimators=_N_ESTIMATORS)[
                "pr_auc"
            ]
        )
    return float(np.mean(scores))


def _cross_condition(df: pl.DataFrame) -> dict:
    """Train on one condition and test on the other (both directions)."""
    feats = feature_cols(df)
    out: dict = {}
    for train_cond, test_cond in [("HOT", "COLD"), ("COLD", "HOT")]:
        train = df.filter(pl.col(CONDITION) == train_cond)
        test = df.filter(pl.col(CONDITION) == test_cond)
        x_tr = train.select(feats).cast(pl.Float32).to_numpy()
        y_tr = train[ACTUAL].to_numpy()
        x_te = test.select(feats).cast(pl.Float32).to_numpy()
        y_te = test[ACTUAL].to_numpy()
        x_tr, y_tr = cap_rows(x_tr, y_tr, _ROW_CAP)
        out[f"{train_cond}->{test_cond}"] = fit_xgb_eval(
            x_tr, y_tr, x_te, y_te, n_estimators=_N_ESTIMATORS
        )
    return out


def _add_lags(
    df: pl.DataFrame, prim: list[str], lags: tuple[int, ...] = (1, 2)
) -> pl.DataFrame:
    """Add lag-``k`` columns for each primitive feature, per experiment."""
    out = df
    for lag in lags:
        out = out.with_columns(
            pl.col(c).shift(lag).over(EXPERIMENT).alias(f"{c}_lag{lag}") for c in prim
        )
    return out.drop_nulls()


def _temporal_features(df: pl.DataFrame) -> dict:
    """Baseline vs. +lag features, evaluated on the fixed train/test split."""
    train, test, feats = split_by_test_experiments(df)
    prim = _primitive_cols(feats)
    x_test = test.select(feats).cast(pl.Float32).to_numpy()
    y_test = test[ACTUAL].to_numpy()

    base = {
        "n_features": len(feats),
        **fit_xgb_eval(
            train.select(feats).cast(pl.Float32).to_numpy(),
            train[ACTUAL].to_numpy(),
            x_test,
            y_test,
            n_estimators=_N_ESTIMATORS,
        ),
    }

    train_lag = _add_lags(train, prim)
    test_lag = _add_lags(test, prim)
    lag_feats = feature_cols(train_lag)
    with_lag = {
        "n_features": len(lag_feats),
        **fit_xgb_eval(
            train_lag.select(lag_feats).cast(pl.Float32).to_numpy(),
            train_lag[ACTUAL].to_numpy(),
            test_lag.select(lag_feats).cast(pl.Float32).to_numpy(),
            test_lag[ACTUAL].to_numpy(),
            n_estimators=_N_ESTIMATORS,
        ),
    }
    return {"baseline": base, "with_lags": with_lag}


def main() -> None:
    """Run the improvement experiments and write the report."""
    df = add_experiment_parts(load_features_labels())
    report = {
        "grouped_cv_pr_auc": {
            "by_experiment": _grouped_cv(df, EXPERIMENT),
            "by_animal": _grouped_cv(df, ANIMAL),
        },
        "cross_condition": _cross_condition(df),
        "temporal_features": _temporal_features(df),
    }
    write_json(report, EDA_OUT_DIR / "experiments.json")
    print(f"grouped CV PR-AUC: {report['grouped_cv_pr_auc']}")  # noqa: T201
    print("cross-condition:")  # noqa: T201
    for k, v in report["cross_condition"].items():
        print(f"  {k}: pr_auc={v['pr_auc']:.3f}")  # noqa: T201
    print("temporal features:")  # noqa: T201
    for k, v in report["temporal_features"].items():
        print(f"  {k}: pr_auc={v['pr_auc']:.3f} (n_features={v['n_features']})")  # noqa: T201


if __name__ == "__main__":
    main()
