"""Keypoint-quality investigation (pipeline hypotheses a/b).

Uses the preprocessed DeepLabCut keypoints to answer:

- how well is each bodypart tracked (likelihood, and frame-to-frame jitter)?
- can we separate *genuine occlusion* from *confident bad tracking*?
- does keypoint quality degrade during rearing (FR=1) or in COLD vs HOT?
- do model errors (FN/FP) concentrate on low-quality frames?
- is the arena-marker floor reference actually used (or silently falling back)?

Output is written as JSON to ``data/front-rear/eda/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from behavysis.constants import EXPERIMENT, FRAME, PRED

from .common import (
    ACTUAL,
    CONDITION,
    EDA_OUT_DIR,
    LABELS_DIR,
    load_model_eval,
    write_json,
)

KEYPOINTS_DIR = Path("data/front-rear/training_data/4_preprocessed")

PCUTOFF = 0.6
TELEPORT_PX = 100.0

# Bodyparts driving the rearing features R01-R08.
FEATURE_BODYPARTS = [
    "nose",
    "ear_l",
    "ear_r",
    "mid_back",
    "lower_back",
    "front_toe_l",
    "front_toe_r",
    "hind_toe_l",
    "hind_toe_r",
]


def _load_kp(fp: Path) -> pl.DataFrame:
    """Load one keypoints file, rat individual only, sorted by frame."""
    return (
        pl.read_parquet(fp)
        .filter(pl.col("individual") == "rat")
        .sort([FRAME, "bodypart"])
    )


def _displacement(sub: pl.DataFrame) -> np.ndarray:
    """Frame-to-frame Euclidean displacement (pixels) for a bodypart."""
    x = sub["x"].to_numpy()
    y = sub["y"].to_numpy()
    return np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0]))


def _per_bodypart_quality(kp: pl.DataFrame) -> pl.DataFrame:
    """Per-bodypart likelihood, displacement and teleport statistics."""
    rows: list[dict] = []
    for sub in kp.partition_by(["bodypart"], maintain_order=True):
        bp = sub["bodypart"][0]
        lik = sub["likelihood"].to_numpy()
        disp = _displacement(sub)
        low_conf = lik < PCUTOFF
        teleport = disp > TELEPORT_PX
        rows.append(
            {
                "bodypart": bp,
                "mean_lik": float(np.mean(lik)),
                "frac_low_conf": float(np.mean(low_conf)),
                "frac_teleport": float(np.mean(teleport)),
                "frac_confident_teleport": float(np.mean(teleport & ~low_conf)),
            }
        )
    return pl.DataFrame(rows)


def _occlusion_run_lengths(kp: pl.DataFrame) -> dict:
    """Median/max consecutive low-likelihood run length for key bodyparts."""
    out: dict = {}
    for bp in FEATURE_BODYPARTS:
        sub = kp.filter(pl.col("bodypart") == bp).sort(FRAME)
        low = (sub["likelihood"].to_numpy() < PCUTOFF).astype(int)
        runs: list[int] = []
        cur = 0
        for v in low:
            if v == 1:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        runs_arr = np.array(runs) if runs else np.array([0])
        out[bp] = {
            "median_run": float(np.median(runs_arr)),
            "p90_run": float(np.percentile(runs_arr, 90)),
            "max_run": int(runs_arr.max()),
        }
    return out


def _arena_check() -> dict:
    """Which individual owns the arena markers (floor reference)."""
    kp = pl.read_parquet(sorted(KEYPOINTS_DIR.glob("*.parquet"))[0])
    arena = kp.filter(pl.col("bodypart").is_in(["arena_l", "arena_r"]))
    owners = arena.group_by("individual").len().to_dicts()
    return {
        "arena_marker_owners": owners,
        "usable_by_floor": any(o["individual"] == "rat" for o in owners),
    }


def _label_breakdown() -> dict:
    """Mean likelihood on FR=0 vs FR=1 frames, averaged over experiments."""
    lik0: list[float] = []
    lik1: list[float] = []
    for fp in sorted(KEYPOINTS_DIR.glob("*.parquet")):
        kp = _load_kp(fp)
        lab = pl.read_parquet(LABELS_DIR / f"{fp.stem}.parquet").select([FRAME, "FR"])
        merged = kp.join(lab, on=FRAME, how="inner")
        grouped = {
            r["FR"]: r["likelihood"]
            for r in merged.group_by("FR").agg(pl.col("likelihood").mean()).to_dicts()
        }
        if 0 in grouped:
            lik0.append(grouped[0])
        if 1 in grouped:
            lik1.append(grouped[1])
    return {
        "mean_lik_FR0": float(np.mean(lik0)),
        "mean_lik_FR1": float(np.mean(lik1)),
        "n_exp_FR1": len(lik1),
    }


def _error_vs_quality() -> dict:
    """Mean per-frame keypoint likelihood by TP/FN/TN/FP (test set)."""
    test_exps = set(load_model_eval("xgb", "test")[EXPERIMENT].unique().to_list())
    eval_df = load_model_eval("xgb", "test").with_columns(
        pl.col(ACTUAL).cast(pl.Int64).alias("y"),
        pl.col(PRED).cast(pl.Int64).alias("p"),
    )
    rows: list[dict] = []
    for exp in sorted(test_exps):
        kp = _load_kp(KEYPOINTS_DIR / f"{exp}.parquet")
        frame_lik = kp.group_by(FRAME).agg(
            pl.col("likelihood").mean().alias("mean_lik")
        )
        merged = eval_df.filter(pl.col(EXPERIMENT) == exp).join(
            frame_lik, on=FRAME, how="inner"
        )
        for y, p, label in [(1, 1, "TP"), (1, 0, "FN"), (0, 0, "TN"), (0, 1, "FP")]:
            sub = merged.filter((pl.col("y") == y) & (pl.col("p") == p))
            rows.append(
                {
                    "cell": label,
                    "n": sub.height,
                    "mean_lik": float(sub["mean_lik"].mean()),
                }
            )
    agg = (
        pl.DataFrame(rows)
        .group_by("cell")
        .agg(pl.col("n").sum(), pl.col("mean_lik").mean())
    )
    return agg.to_dicts()


def _condition_quality() -> dict:
    """Mean likelihood per condition."""
    rows: list[dict] = []
    for fp in sorted(KEYPOINTS_DIR.glob("*.parquet")):
        kp = _load_kp(fp)
        rows.append(
            {
                CONDITION: fp.stem.split("-")[2],
                "mean_lik": float(kp["likelihood"].mean()),
            }
        )
    return pl.DataFrame(rows).group_by(CONDITION).mean().to_dicts()


def main() -> None:
    """Run keypoint-quality analysis and write the report."""
    # Per-bodypart quality averaged over all experiments.
    accum: dict[str, dict[str, float]] = {}
    for fp in sorted(KEYPOINTS_DIR.glob("*.parquet")):
        for r in _per_bodypart_quality(_load_kp(fp)).to_dicts():
            bp = r.pop("bodypart")
            acc = accum.setdefault(bp, dict.fromkeys(r, 0.0))
            for k, v in r.items():
                acc[k] += v
    n = len(list(KEYPOINTS_DIR.glob("*.parquet")))
    bp_df = pl.DataFrame(
        {"bodypart": k, **{kk: vv / n for kk, vv in v.items()}}
        for k, v in accum.items()
    ).sort("mean_lik")

    report = {
        "teleport_px": TELEPORT_PX,
        "pcutoff": PCUTOFF,
        "per_bodypart_quality": bp_df.to_dicts(),
        "occlusion_run_lengths": _occlusion_run_lengths(
            _load_kp(sorted(KEYPOINTS_DIR.glob("*.parquet"))[0])
        ),
        "label_breakdown": _label_breakdown(),
        "error_vs_quality": _error_vs_quality(),
        "condition_quality": _condition_quality(),
        "arena_check": _arena_check(),
    }
    write_json(report, EDA_OUT_DIR / "keypoint_quality.json")
    print("per-bodypart quality (worst first):")  # noqa: T201
    for r in bp_df.head(10).to_dicts():
        print(  # noqa: T201
            f"  {r['bodypart']}: mean_lik={r['mean_lik']:.2f} "
            f"low_conf={r['frac_low_conf']:.2f} teleport={r['frac_teleport']:.3f} "
            f"confident_teleport={r['frac_confident_teleport']:.3f}"
        )
    print(f"arena_check: {report['arena_check']}")  # noqa: T201
    print(f"label_breakdown (lik): {report['label_breakdown']}")  # noqa: T201
    print(f"error_vs_quality: {report['error_vs_quality']}")  # noqa: T201
    print(f"condition_quality: {report['condition_quality']}")  # noqa: T201


if __name__ == "__main__":
    main()
