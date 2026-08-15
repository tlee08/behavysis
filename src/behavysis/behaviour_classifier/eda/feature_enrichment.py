"""Feature-enrichment investigation for front-rearing.

Re-extracts features from the preprocessed keypoints with three fixes over
the production ``extract_rearing`` pipeline, plus new candidate features, and
measures their signal (univariate AUROC) and downstream classification
benefit.

Fixes applied here (for evaluation, before promotion to production):

1. arena-marker floor (markers live under the ``"single"`` individual);
2. likelihood gating -- positions are set to NaN when ``likelihood < pcutoff``
   and NaN propagates through features (no blind forward-fill smearing);
3. "front" anchored on nose/ears (reliably tracked) instead of front toes.

Output is written to ``data/front-rear/eda/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

from .common import EDA_OUT_DIR, load_features_labels, write_json

KEYPOINTS_DIR = Path("data/front-rear/training_data/4_preprocessed")
FPS = 50.0
PX_PER_MM = 4.65
PCUTOFF = 0.6
_MIN_VALID = 100

HEAD_BPTS = ["nose", "ear_l", "ear_r"]
HIND_BPTS = [
    "hind_toe_l",
    "hind_toe_r",
    "hind_heel_l",
    "hind_heel_r",
    "hind_knee_l",
    "hind_knee_r",
]
FRONT_BPTS = ["front_toe_l", "front_toe_r", "front_knee_l", "front_knee_r"]

ROLL_WINDOWS = [3, 6, 12, 25, 33, 50]


def _load_kp(fp: Path) -> pl.DataFrame:
    """Load one keypoints file (rat individual), sorted by frame."""
    return (
        pl.read_parquet(fp)
        .filter(pl.col("individual") == "rat")
        .sort(["frame", "bodypart"])
    )


def _arena_floor(fp: Path) -> float:
    """Constant floor y from the arena markers (individual 'single')."""
    arena = pl.read_parquet(fp).filter(pl.col("individual") == "single").select("y")
    if arena.height == 0:
        return float("nan")
    return float(arena["y"].median())


def _bodypart_arrays(
    kp: pl.DataFrame, bpts: list[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {bodypart: (x, y)} with NaN where likelihood < pcutoff."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for bp in bpts:
        sub = kp.filter(pl.col("bodypart") == bp).sort("frame")
        x = sub["x"].to_numpy().astype(np.float64)
        y = sub["y"].to_numpy().astype(np.float64)
        lik = sub["likelihood"].to_numpy().astype(np.float64)
        mask = lik < PCUTOFF
        out[bp] = (np.where(mask, np.nan, x), np.where(mask, np.nan, y))
    return out


def _nanmean_y(
    bps: dict[str, tuple[np.ndarray, np.ndarray]], names: list[str]
) -> np.ndarray:
    """NaN-aware mean y over a set of bodyparts."""
    with np.errstate(all="ignore"):
        return np.nanmean(np.column_stack([bps[n][1] for n in names]), axis=1)


def _nanmean_x(
    bps: dict[str, tuple[np.ndarray, np.ndarray]], names: list[str]
) -> np.ndarray:
    """NaN-aware mean x over a set of bodyparts."""
    with np.errstate(all="ignore"):
        return np.nanmean(np.column_stack([bps[n][0] for n in names]), axis=1)


def _nan_roll(arr: np.ndarray, w: int, stat: str) -> np.ndarray:
    """NaN-aware centred rolling statistic (pandas)."""
    s = pd.Series(arr)
    if stat == "mean":
        return s.rolling(w, center=True, min_periods=1).mean().to_numpy()
    if stat == "std":
        return s.rolling(w, center=True, min_periods=1).std().to_numpy()
    if stat == "min":
        return s.rolling(w, center=True, min_periods=1).min().to_numpy()
    return s.rolling(w, center=True, min_periods=1).max().to_numpy()


def _angle_deg(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Angle from horizontal in degrees (NaN-propagating)."""
    return np.degrees(np.arctan2(-dy, dx))


def _vel(y: np.ndarray, px_per_mm: float, fps: float) -> np.ndarray:
    """Vertical velocity (mm/s), NaN-aware, with light smoothing."""
    ys = _nan_roll(y, 3, "mean")
    return -np.diff(ys, prepend=ys[0]) / px_per_mm * fps


def _primitives(kp: pl.DataFrame, floor: float) -> dict[str, np.ndarray]:
    """Compute fixed + new primitive features (NaN-propagating)."""
    all_bpts = (
        HEAD_BPTS
        + HIND_BPTS
        + FRONT_BPTS
        + ["mid_back", "lower_back", "tail_base", "tail_tip"]
    )
    b = _bodypart_arrays(kp, all_bpts)

    head_x = _nanmean_x(b, HEAD_BPTS)
    head_y = _nanmean_y(b, HEAD_BPTS)
    nose_y = b["nose"][1]
    ear_y = _nanmean_y(b, ["ear_l", "ear_r"])
    hind_y = _nanmean_y(b, HIND_BPTS)
    front_toe_y = _nanmean_y(b, ["front_toe_l", "front_toe_r"])
    hind_toe_y = _nanmean_y(b, ["hind_toe_l", "hind_toe_r"])
    lower_back_y = b["lower_back"][1]
    lower_back_x = b["lower_back"][0]
    mid_back_y = b["mid_back"][1]
    mid_back_x = b["mid_back"][0]

    f: dict[str, np.ndarray] = {}
    # -- existing, fixed --
    f["R01_back_angle"] = _angle_deg(
        lower_back_x - mid_back_x, lower_back_y - mid_back_y
    )
    f["R02_head_elevation"] = (floor - head_y) / PX_PER_MM
    f["R03_body_elongation"] = (hind_y - head_y) / (
        np.abs(head_x - _nanmean_x(b, HIND_BPTS)) + 1e-6
    )
    f["R05_front_paw_elevation"] = (hind_toe_y - front_toe_y) / PX_PER_MM
    f["R06_head_vy"] = _vel(head_y, PX_PER_MM, FPS)
    f["R07_whole_body_angle"] = _angle_deg(
        head_x - _nanmean_x(b, HIND_BPTS), head_y - hind_y
    )
    f["R08_upper_body_angle"] = _angle_deg(head_x - lower_back_x, head_y - lower_back_y)
    # centroid vertical velocity
    with np.errstate(all="ignore"):
        all_y = np.nanmean(np.column_stack([b[n][1] for n in all_bpts]), axis=1)
    f["R04_centroid_vy"] = _vel(all_y, PX_PER_MM, FPS)
    # -- new candidates --
    f["N1_nose_elevation"] = (floor - nose_y) / PX_PER_MM
    f["N2_ear_elevation"] = (floor - ear_y) / PX_PER_MM
    f["N3_head_pitch"] = (ear_y - nose_y) / PX_PER_MM
    f["N4_hind_ground"] = (floor - hind_toe_y) / PX_PER_MM
    f["N5_verticality_ratio"] = (floor - head_y) / (np.abs(floor - hind_y) + 1e-6)
    f["N6_paw_asymmetry"] = (
        np.abs(b["front_toe_l"][1] - b["front_toe_r"][1]) / PX_PER_MM
    )
    f["N7_n_low_conf"] = np.nansum(
        np.column_stack(
            [
                np.isnan(b[n][1])
                for n in HEAD_BPTS + FRONT_BPTS + ["lower_back", "mid_back"]
            ]
        ),
        axis=1,
    )
    f["N8_head_speed"] = np.hypot(
        _vel(head_x, PX_PER_MM, FPS), _vel(head_y, PX_PER_MM, FPS)
    )
    f["N9_back_angle_abs"] = np.abs(f["R01_back_angle"])
    return f


def _rolling(primitives: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Rolling mean/std/min/max for every primitive."""
    out: dict[str, np.ndarray] = {}
    for name, arr in primitives.items():
        for w in ROLL_WINDOWS:
            for stat in ("mean", "std", "min", "max"):
                out[f"{name}_{stat}_w{w}"] = _nan_roll(arr, w, stat)
    return out


def extract_one(fp: Path) -> pl.DataFrame:
    """Extract primitives + rolling features for one experiment."""
    kp = _load_kp(fp)
    floor = _arena_floor(fp)
    prim = _primitives(kp, floor)
    feats = prim | _rolling(prim)
    frames = kp["frame"].unique().sort().to_numpy()
    data = {"frame": frames.astype(np.int64)}
    data |= {k: v.astype(np.float32) for k, v in feats.items()}
    return pl.DataFrame(data)


def _univariate_auc(feats_df: pl.DataFrame, y: np.ndarray) -> pl.DataFrame:
    """Signed AUROC of each feature against the FR label."""
    rows = []
    for c in feats_df.columns:
        if c == "frame":
            continue
        col = feats_df[c].to_numpy()
        valid = np.isfinite(col)
        if valid.sum() < _MIN_VALID:
            rows.append({"feature": c, "auc": 0.5, "valid_frac": 0.0})
            continue
        auc = roc_auc_score(y[valid], col[valid])
        rows.append(
            {"feature": c, "auc": max(auc, 1 - auc), "valid_frac": valid.mean()}
        )
    return pl.DataFrame(rows).sort("auc", descending=True)


def main() -> None:
    """Extract features for all experiments and evaluate signal."""
    fps = sorted(KEYPOINTS_DIR.glob("*.parquet"))
    # Re-extract all experiments and join with labels.
    frames = [
        extract_one(fp).with_columns(pl.lit(fp.stem).alias("experiment")) for fp in fps
    ]
    all_feats = pl.concat(frames, how="diagonal_relaxed")

    labels = load_features_labels().select(["experiment", "frame", "actual"])
    merged = all_feats.join(labels, on=["experiment", "frame"], how="inner")
    auc_df = _univariate_auc(
        merged.select(
            [c for c in merged.columns if c not in ("frame", "experiment", "actual")]
        ),
        merged["actual"].to_numpy(),
    )

    merged.write_parquet(EDA_OUT_DIR / "enriched_features.parquet")

    report = {
        "n_features": auc_df.height,
        "top_30_univariate_auc": auc_df.head(30).to_dicts(),
    }
    write_json(report, EDA_OUT_DIR / "feature_enrichment_auc.json")
    auc_df.write_parquet(EDA_OUT_DIR / "feature_enrichment_auc.parquet")
    print("top features by univariate AUROC:")  # noqa: T201
    for r in auc_df.head(20).to_dicts():
        print(f"  {r['feature']}: auc={r['auc']:.3f} valid={r['valid_frac']:.2f}")  # noqa: T201


if __name__ == "__main__":
    main()
