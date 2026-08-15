"""Re-extract features with the production NaN-gated extractor and evaluate.

Extracts rearing features for every experiment using ``rearing_compute``
(which now likelihood-gates positions and includes R09-R17), saves them,
and compares classification against the stored (pre-fix) features.

Output is written to ``data/front-rear/eda/``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from behavysis.constants import EXPERIMENT
from behavysis.funcs.extract_features.extract_rearing import rearing_compute

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

KEYPOINTS_DIR = Path("data/front-rear/training_data/4_preprocessed")
FPS = 50.0
PX_PER_MM = 4.65
PCUTOFF = 0.6

_ROW_CAP = 250_000
_N_ESTIMATORS = 120


def _extract_all() -> pl.DataFrame:
    """Extract features for every experiment and join labels."""
    pieces = []
    for fp in sorted(KEYPOINTS_DIR.glob("*.parquet")):
        kp = pl.read_parquet(fp)
        feats = rearing_compute(kp, fps=FPS, px_per_mm=PX_PER_MM, pcutoff=PCUTOFF)
        pieces.append(feats.with_columns(pl.lit(fp.stem).alias(EXPERIMENT)))
    return pl.concat(pieces, how="diagonal_relaxed")


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
    """Extract, save, and evaluate."""
    test_exps = set(load_model_eval("xgb", "test")[EXPERIMENT].unique().to_list())

    baseline = load_features_labels()
    base_feats = feature_cols(baseline)

    prod = _extract_all().join(
        load_features_labels().select([EXPERIMENT, "frame", ACTUAL]),
        on=[EXPERIMENT, "frame"],
        how="inner",
    )
    prod.write_parquet(EDA_OUT_DIR / "production_features.parquet")
    prod_feats = [c for c in prod.columns if c not in (EXPERIMENT, "frame", ACTUAL)]

    report = {
        "baseline_stored": _eval(baseline, base_feats, test_exps),
        "production_nan_gated": _eval(prod, prod_feats, test_exps),
    }
    write_json(report, EDA_OUT_DIR / "re_extract_eval.json")
    for name, res in report.items():
        print(  # noqa: T201
            f"{name}: pr_auc={res['pr_auc']:.3f} roc_auc={res['roc_auc']:.3f} "
            f"n_features={res['n_features']}"
        )


if __name__ == "__main__":
    main()
