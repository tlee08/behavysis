"""Train/test split analysis and model-agreement diagnostics.

Quantifies the structure of the held-out-video split and whether the
generalisation gap is data- or model-driven:

- experiment / animal / condition / batch overlap between train and test
- per-video and per-condition AUPRC/ROC for both models
- how much the two models agree (probability correlation, shared misses)

Output is written as JSON to ``data/front-rear/eda/``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from behavysis.constants import EXPERIMENT, FRAME, PRED, PROB

from .common import (
    ACTUAL,
    ANIMAL,
    BATCH,
    CAMERA,
    CONDITION,
    EDA_OUT_DIR,
    MODELS,
    add_experiment_parts,
    load_model_eval,
    per_group_metrics,
    pooled_metrics,
    write_json,
)

SPLIT = "split"
MODEL = "model"


def _build_df() -> pl.DataFrame:
    """Load every model/split eval parquet into one long frame."""
    pieces = [
        load_model_eval(model, split).with_columns(
            pl.lit(split).alias(SPLIT), pl.lit(model).alias(MODEL)
        )
        for split in ["train", "test"]
        for model in MODELS
    ]
    return add_experiment_parts(pl.concat(pieces))


def _overlap(df: pl.DataFrame) -> dict:
    """Count how many train/test experiments share each identity attribute."""
    out: dict = {}
    for col in [ANIMAL, CONDITION, BATCH, CAMERA]:
        train_vals = set(df.filter(pl.col(SPLIT) == "train")[col].unique().to_list())
        test_vals = set(df.filter(pl.col(SPLIT) == "test")[col].unique().to_list())
        out[col] = {
            "n_train": len(train_vals),
            "n_test": len(test_vals),
            "overlap": len(train_vals & test_vals),
        }
    return out


def _metrics(df: pl.DataFrame) -> dict:
    """Pooled metrics per (split, model) plus per-video/per-condition detail."""
    out: dict = {}
    for split in ["train", "test"]:
        out[split] = {}
        for model in MODELS:
            sub = df.filter((pl.col(SPLIT) == split) & (pl.col(MODEL) == model))
            out[split][model] = {"pooled": pooled_metrics(sub)}
        test_xgb = df.filter((pl.col(SPLIT) == "test") & (pl.col(MODEL) == "xgb"))
        out[split]["per_video"] = (
            per_group_metrics(test_xgb, [EXPERIMENT]).sort("pr_auc").to_dicts()
        )
        out[split]["per_condition"] = per_group_metrics(
            test_xgb, [CONDITION]
        ).to_dicts()
    return out


def _model_agreement(df: pl.DataFrame) -> dict:
    """Correlation and shared-miss statistics between the two models (test)."""
    test = df.filter(pl.col(SPLIT) == "test")
    wide = (
        test.filter(pl.col(MODEL) == "xgb")
        .select(
            [
                EXPERIMENT,
                FRAME,
                ACTUAL,
                pl.col(PROB).alias("prob_xgb"),
                pl.col(PRED).alias("pred_xgb"),
            ]
        )
        .join(
            test.filter(pl.col(MODEL) == "tabpfn").select(
                [
                    EXPERIMENT,
                    FRAME,
                    pl.col(PROB).alias("prob_tab"),
                    pl.col(PRED).alias("pred_tab"),
                ]
            ),
            on=[EXPERIMENT, FRAME],
            how="inner",
        )
    )
    pos = wide.filter(pl.col(ACTUAL) == 1)
    both_miss = pos.filter((pl.col("pred_xgb") == 0) & (pl.col("pred_tab") == 0)).height

    ap_xgb = per_group_metrics(
        wide.select([EXPERIMENT, ACTUAL, pl.col("prob_xgb").alias(PROB)]), [EXPERIMENT]
    ).select([EXPERIMENT, pl.col("pr_auc").alias("ap_xgb")])
    ap_tab = per_group_metrics(
        wide.select([EXPERIMENT, ACTUAL, pl.col("prob_tab").alias(PROB)]), [EXPERIMENT]
    ).select([EXPERIMENT, pl.col("pr_auc").alias("ap_tab")])
    ap = ap_xgb.join(ap_tab, on=EXPERIMENT)

    return {
        "prob_corr": float(np.corrcoef(wide["prob_xgb"], wide["prob_tab"])[0, 1]),
        "per_video_ap_corr": float(np.corrcoef(ap["ap_xgb"], ap["ap_tab"])[0, 1]),
        "n_pos_frames": pos.height,
        "both_miss_pos_frames": both_miss,
        "both_miss_frac": both_miss / pos.height if pos.height else float("nan"),
    }


def main() -> None:
    """Run split and agreement analysis and write the report."""
    df = _build_df()
    report = {
        "overlap": _overlap(df),
        "metrics": _metrics(df),
        "model_agreement": _model_agreement(df),
    }
    write_json(report, EDA_OUT_DIR / "split_analysis.json")

    agree = report["model_agreement"]
    print(  # noqa: T201
        f"prob_corr={agree['prob_corr']:.3f} "
        f"per_video_ap_corr={agree['per_video_ap_corr']:.3f} "
        f"both_miss_frac={agree['both_miss_frac']:.3f}"
    )
    for split in ["train", "test"]:
        for model in MODELS:
            m = report["metrics"][split][model]["pooled"]
            print(  # noqa: T201
                f"[{split}/{model}] pr_auc={m['pr_auc']:.3f} "
                f"roc_auc={m['roc_auc']:.3f} pos_rate={m['pos_rate']:.3f}"
            )
    for row in report["metrics"]["test"]["per_condition"]:
        print(  # noqa: T201
            f"[test/{row[CONDITION]}] pr_auc={row['pr_auc']:.3f} n_pos={row['n_pos']}"
        )


if __name__ == "__main__":
    main()
