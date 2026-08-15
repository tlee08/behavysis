"""Data-integrity census for the classifier's training data.

Answers, from first principles, "what do we actually have?":

- are the feature/label parquet files complete (valid footer)?
- how many experiments, frames, positive frames, and positive bouts exist?
- how redundant are frames within a bout (effective sample size)?
- are frames contiguous, and do features contain NaN / inf?

Output is printed and written as JSON to ``data/front-rear/eda/``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from behavysis.constants import BOUT_ID, EXPERIMENT, FRAME

from .common import (
    ACTUAL,
    CONDITION,
    EDA_OUT_DIR,
    FEATURES_DIR,
    LABELS_DIR,
    add_experiment_parts,
    feature_cols,
    load_features_labels,
    write_json,
)

if TYPE_CHECKING:
    from pathlib import Path

_PAR1 = b"PAR1"


def _check_footer(fp: Path) -> bool:
    """Return True if ``fp`` ends with the parquet magic bytes."""
    with fp.open("rb") as fh:
        fh.seek(-4, 2)
        return fh.read(4) == _PAR1


def _integrity() -> dict:
    """Check every feature and label parquet footer."""
    feats = sorted(FEATURES_DIR.glob("*.parquet"))
    labels = sorted(LABELS_DIR.glob("*.parquet"))
    return {
        "features": {
            "total": len(feats),
            "valid": sum(_check_footer(f) for f in feats),
        },
        "labels": {
            "total": len(labels),
            "valid": sum(_check_footer(f) for f in labels),
        },
    }


def _census(df: pl.DataFrame) -> dict:
    """Frame, bout and experiment-level counts."""
    bouts = df.filter(pl.col(ACTUAL) == 1)
    bout_lens = bouts.group_by(BOUT_ID).len().sort("len")["len"]
    return {
        "n_experiments": df[EXPERIMENT].n_unique(),
        "n_frames": df.height,
        "n_pos_frames": bouts.height,
        "pos_rate": bouts.height / df.height,
        "n_pos_bouts": bouts[BOUT_ID].n_unique(),
        "bout_len_mean": float(bout_lens.mean()),
        "bout_len_median": float(bout_lens.median()),
        "bout_len_p5": float(bout_lens.quantile(0.05)),
        "bout_len_p95": float(bout_lens.quantile(0.95)),
        "bout_len_max": int(bout_lens.max()),
        "frames_per_bout_ratio": df.height / bouts[BOUT_ID].n_unique(),
    }


def _contiguity(df: pl.DataFrame) -> dict:
    """Check that each experiment's frames are contiguous from its min frame."""
    gaps = (
        df.sort([EXPERIMENT, FRAME])
        .with_columns(pl.col(FRAME).diff().over(EXPERIMENT).alias("diff"))
        .filter(pl.col("diff").is_not_null() & (pl.col("diff") != 1))
        .height
    )
    return {"n_frame_gaps": gaps}


def _feature_health(df: pl.DataFrame) -> dict:
    """NaN / inf counts across all feature columns."""
    x = df.select(feature_cols(df)).cast(pl.Float64).to_numpy()
    return {"n_nan": int(np.isnan(x).sum()), "n_inf": int(np.isinf(x).sum())}


def _per_condition(df: pl.DataFrame) -> dict:
    """Positive rate and bout count per HOT/COLD condition."""
    out: dict = {}
    for sub in df.partition_by([CONDITION], maintain_order=True):
        key = sub[CONDITION][0]
        bouts = sub.filter(pl.col(ACTUAL) == 1)
        out[key] = {
            "n_experiments": sub[EXPERIMENT].n_unique(),
            "n_frames": sub.height,
            "n_pos_bouts": bouts[BOUT_ID].n_unique(),
            "pos_rate": bouts.height / sub.height,
        }
    return out


def main() -> None:
    """Run the census and write the report."""
    df = add_experiment_parts(load_features_labels())
    report = {
        "integrity": _integrity(),
        "census": _census(df),
        "contiguity": _contiguity(df),
        "feature_health": _feature_health(df),
        "per_condition": _per_condition(df),
    }
    write_json(report, EDA_OUT_DIR / "data_integrity.json")
    print(  # noqa: T201
        f"n_experiments={report['census']['n_experiments']} "
        f"n_frames={report['census']['n_frames']} "
        f"pos_rate={report['census']['pos_rate']:.3f} "
        f"n_pos_bouts={report['census']['n_pos_bouts']} "
        f"bout_len_median={report['census']['bout_len_median']:.0f} "
        f"frames_per_bout={report['census']['frames_per_bout_ratio']:.0f} "
        f"feature_health={report['feature_health']}"
    )


if __name__ == "__main__":
    main()
