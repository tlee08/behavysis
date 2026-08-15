"""Sample-efficiency experiments.

Quantifies how much training data is actually needed, since frames within a
behavioural bout are near-duplicates:

- learning curve by number of training videos (pooled test PR-AUC)
- stride scan: train on every 1/2/4/8-th frame
- negative-ratio scan: keep all positives, subsample negatives to a ratio

Output is written as JSON to ``data/front-rear/eda/``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from behavysis.constants import EXPERIMENT

from .common import (
    ACTUAL,
    EDA_OUT_DIR,
    cap_rows,
    fit_xgb_eval,
    load_features_labels,
    split_by_test_experiments,
    write_json,
)

_ROW_CAP = 250_000
_N_ESTIMATORS = 120


def _video_learning_curve(df: pl.DataFrame) -> dict:
    """Test PR-AUC vs. number of training videos."""
    train, test, feats = split_by_test_experiments(df)
    x_test = test.select(feats).cast(pl.Float32).to_numpy()
    y_test = test[ACTUAL].to_numpy()
    train_exps = train[EXPERIMENT].unique().to_list()

    curve: dict = {}
    rng = np.random.default_rng(42)
    for n_videos in [8, 16, 32, 64, 103]:
        exps = rng.choice(
            train_exps, size=min(n_videos, len(train_exps)), replace=False
        )
        sub = train.filter(pl.col(EXPERIMENT).is_in(exps))
        x = sub.select(feats).cast(pl.Float32).to_numpy()
        y = sub[ACTUAL].to_numpy()
        x, y = cap_rows(x, y, _ROW_CAP)
        curve[str(n_videos)] = fit_xgb_eval(
            x, y, x_test, y_test, n_estimators=_N_ESTIMATORS
        )
    return curve


def _stride_scan(df: pl.DataFrame) -> dict:
    """Test PR-AUC when training on every ``stride``-th frame."""
    train, test, feats = split_by_test_experiments(df)
    x_test = test.select(feats).cast(pl.Float32).to_numpy()
    y_test = test[ACTUAL].to_numpy()

    out: dict = {}
    for stride in [1, 2, 4, 8]:
        sub = train.filter(pl.col("frame") % stride == 0)
        x = sub.select(feats).cast(pl.Float32).to_numpy()
        y = sub[ACTUAL].to_numpy()
        x, y = cap_rows(x, y, _ROW_CAP)
        out[str(stride)] = fit_xgb_eval(
            x, y, x_test, y_test, n_estimators=_N_ESTIMATORS
        )
    return out


def _negative_ratio_scan(df: pl.DataFrame) -> dict:
    """Test PR-AUC when negatives are subsampled to ``pos:neg`` ratio."""
    train, test, feats = split_by_test_experiments(df)
    x_test = test.select(feats).cast(pl.Float32).to_numpy()
    y_test = test[ACTUAL].to_numpy()

    pos = train.filter(pl.col(ACTUAL) == 1)
    neg = train.filter(pl.col(ACTUAL) == 0)
    out: dict = {}
    rng = np.random.default_rng(42)
    for neg_per_pos in [1.0, 4.0, 16.0, 100.0]:
        n_neg = int(len(pos) * neg_per_pos)
        neg_sub = neg.gather(
            rng.choice(len(neg), size=min(n_neg, len(neg)), replace=False)
        )
        sub = pl.concat([pos, neg_sub])
        x = sub.select(feats).cast(pl.Float32).to_numpy()
        y = sub[ACTUAL].to_numpy()
        x, y = cap_rows(x, y, _ROW_CAP)
        out[str(neg_per_pos)] = fit_xgb_eval(
            x, y, x_test, y_test, n_estimators=_N_ESTIMATORS
        )
    return out


def main() -> None:
    """Run sample-efficiency scans and write the report."""
    df = load_features_labels()
    report = {
        "video_learning_curve": _video_learning_curve(df),
        "stride_scan": _stride_scan(df),
        "negative_ratio_scan": _negative_ratio_scan(df),
    }
    write_json(report, EDA_OUT_DIR / "sample_efficiency.json")
    print("video learning curve (n_videos -> test pr_auc):")  # noqa: T201
    for k, v in report["video_learning_curve"].items():
        print(f"  {k}: {v['pr_auc']:.3f}")  # noqa: T201
    print("stride scan (stride -> test pr_auc):")  # noqa: T201
    for k, v in report["stride_scan"].items():
        print(f"  {k}: {v['pr_auc']:.3f}")  # noqa: T201


if __name__ == "__main__":
    main()
