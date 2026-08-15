"""Classification evaluation of enriched features.

Compares, on the fixed held-out-experiment split, the test PR-AUC of:

- ``baseline``  : stored production features (200, ffill'd, buggy floor)
- ``fixed_R``   : re-extracted R01-R08 (fixed floor, NaN-gated) + rolling
- ``enriched``  : fixed_R + new candidate features (N1-N9) + rolling

Output is written as JSON to ``data/front-rear/eda/``.
"""

from __future__ import annotations

import polars as pl

from behavysis.constants import EXPERIMENT

from .common import (
    ACTUAL,
    EDA_OUT_DIR,
    cap_rows,
    feature_cols,
    fit_xgb_eval,
    load_features_labels,
    load_model_eval,
    write_json,
)

_ROW_CAP = 250_000
_N_ESTIMATORS = 120


def _test_experiments() -> set[str]:
    return set(load_model_eval("xgb", "test")[EXPERIMENT].unique().to_list())


def _eval(df: pl.DataFrame, feats: list[str], test_exps: set[str]) -> dict:
    """Train XGB on train experiments and report pooled test metrics."""
    train = df.filter(~pl.col(EXPERIMENT).is_in(test_exps))
    test = df.filter(pl.col(EXPERIMENT).is_in(test_exps))
    x_train = train.select(feats).cast(pl.Float32).to_numpy()
    y_train = train[ACTUAL].to_numpy()
    x_test = test.select(feats).cast(pl.Float32).to_numpy()
    y_test = test[ACTUAL].to_numpy()
    x_train, y_train = cap_rows(x_train, y_train, _ROW_CAP)
    return {
        "n_features": len(feats),
        **fit_xgb_eval(x_train, y_train, x_test, y_test, n_estimators=_N_ESTIMATORS),
    }


def main() -> None:
    """Run the comparison."""
    test_exps = _test_experiments()

    baseline = load_features_labels()
    base_feats = feature_cols(baseline)

    enriched = pl.read_parquet(EDA_OUT_DIR / "enriched_features.parquet")
    enrich_cols = [
        c for c in enriched.columns if c not in (EXPERIMENT, "frame", ACTUAL)
    ]
    r_feats = [c for c in enrich_cols if c.startswith("R0")]

    report = {
        "baseline": _eval(baseline, base_feats, test_exps),
        "fixed_R": _eval(enriched, r_feats, test_exps),
        "enriched": _eval(enriched, enrich_cols, test_exps),
    }
    write_json(report, EDA_OUT_DIR / "feature_enrichment_eval.json")
    for name, res in report.items():
        print(  # noqa: T201
            f"{name}: pr_auc={res['pr_auc']:.3f} roc_auc={res['roc_auc']:.3f} "
            f"n_features={res['n_features']}"
        )


if __name__ == "__main__":
    main()
