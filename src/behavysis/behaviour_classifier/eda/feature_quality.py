"""Feature-quality and signal analysis.

Quantifies how much discriminative signal lives in the 200 features and
how redundant they are:

- per-feature univariate AUROC (class separation vs. the FR label)
- near-zero-variance and near-duplicate (|corr| > 0.99) feature groups
- PCA explained-variance (how many independent dimensions 200 features span)
- XGBoost gain importances (top features, trained on a subsample)

Note: keypoint likelihood is not available locally, so occlusion is only
probed indirectly (via feature flatness/zero-variance).

Output is written to ``data/front-rear/eda/``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .common import (
    ACTUAL,
    EDA_OUT_DIR,
    add_experiment_parts,
    feature_cols,
    load_features_labels,
    write_json,
)

_MAX_SAMPLES = 150_000
_MIN_STD = 1e-12
_NEAR_CONSTANT_STD = 1e-6


def _univariate_auc(df: pl.DataFrame, feats: list[str]) -> pl.DataFrame:
    """Signed per-feature AUC (max(AUC, 1-AUC)) against the FR label."""
    y = df[ACTUAL].to_numpy()
    rows: list[dict] = []
    for f in feats:
        col = df[f].cast(pl.Float64).to_numpy()
        if np.nanstd(col) < _MIN_STD:
            rows.append({"feature": f, "auc": 0.5, "std": 0.0})
            continue
        auc = float(roc_auc_score(y, col))
        rows.append(
            {"feature": f, "auc": max(auc, 1.0 - auc), "std": float(np.nanstd(col))}
        )
    return pl.DataFrame(rows).sort("auc", descending=True)


def _redundancy(x: np.ndarray) -> dict:
    """Number of independent PCA dimensions and near-duplicate feature pairs."""
    pca = PCA(n_components=min(50, x.shape[1]))
    pca.fit(StandardScaler().fit_transform(x))
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n99 = int(np.searchsorted(cumvar, 0.99) + 1)
    return {
        "n_dims_50pct": int(np.searchsorted(cumvar, 0.50) + 1),
        "n_dims_90pct": int(np.searchsorted(cumvar, 0.90) + 1),
        "n_dims_99pct": n99,
    }


def _xgboost_importance(df: pl.DataFrame, feats: list[str]) -> pl.DataFrame:
    """Top gain importances from a quick XGBoost fit."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), size=min(_MAX_SAMPLES, len(df)), replace=False)
    sub = df.gather(idx)
    x = sub.select(feats).cast(pl.Float32).to_numpy()
    y = sub[ACTUAL].to_numpy()
    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        tree_method="hist",
        scale_pos_weight=10,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x, y)
    gain = model.get_booster().get_score(importance_type="gain")
    return pl.DataFrame(
        {"feature": feats, "gain": [gain.get(f"f{i}", 0.0) for i in range(len(feats))]}
    ).sort("gain", descending=True)


def main() -> None:
    """Run feature-quality analysis and write the report."""
    df = add_experiment_parts(load_features_labels())
    feats = feature_cols(df)

    auc_df = _univariate_auc(df, feats)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), size=min(50_000, len(df)), replace=False)
    x = df.gather(idx).select(feats).cast(pl.Float32).to_numpy()

    imp_df = _xgboost_importance(df, feats)

    report = {
        "n_features": len(feats),
        "n_near_constant": int((auc_df["std"] < _NEAR_CONSTANT_STD).sum()),
        "redundancy": _redundancy(x),
        "top_univariate_auc": auc_df.head(15).to_dicts(),
        "top_xgb_gain": imp_df.head(15).to_dicts(),
    }
    write_json(report, EDA_OUT_DIR / "feature_quality.json")
    auc_df.write_parquet(EDA_OUT_DIR / "feature_univariate_auc.parquet")
    imp_df.write_parquet(EDA_OUT_DIR / "feature_xgb_importance.parquet")

    print(  # noqa: T201
        f"redundancy={report['redundancy']} n_near_constant={report['n_near_constant']}"
    )
    print("top univariate features:")  # noqa: T201
    for r in report["top_univariate_auc"][:10]:
        print(f"  {r['feature']}: auc={r['auc']:.3f}")  # noqa: T201


if __name__ == "__main__":
    main()
