"""Generic feature extraction from preprocessed keypoints.

Computes a comprehensive feature battery from any keypoint configuration:
pairwise distances, movements, convex hull, cdist statistics, probability
features, rolling window aggregates, deviations, and percentile ranks.

No semantic bodypart roles — purely programmatic from individuals + bodyparts.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from loguru import logger
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from behavysis.transforms.keypoint import check_bpts_exist

if TYPE_CHECKING:
    from behavysis.constants import Array1D, Array2D
    from behavysis.models import ExperimentConfig, ExperimentMetadata

# ═══════════════════════════════════════════════════════════════════════════════
# Rolling window divisors
# ═══════════════════════════════════════════════════════════════════════════════

ROLL_WINDOW_DIVISORS: list[float] = [
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
    8.0,
    8.5,
    9.0,
    9.5,
    10.0,
    10.5,
    11.0,
    11.5,
    12.0,
    12.5,
    13.0,
    13.5,
    14.0,
    14.5,
    15.0,
]


# ═══════════════════════════════════════════════════════════════════════════════
# Vectorized math helpers (generic, reusable)
# ═══════════════════════════════════════════════════════════════════════════════


def _euclidean(
    ax: Array1D,
    ay: Array1D,
    bx: Array1D,
    by: Array1D,
    px_per_mm: float,
) -> Array1D:
    """Vectorized Euclidean distance between two point sets, scaled to mm."""
    return np.hypot(ax - bx, ay - by) / px_per_mm


def _movement_frame_to_frame(ax: Array1D, ay: Array1D, px_per_mm: float) -> Array1D:
    """Frame-to-frame Euclidean movement for a single body-part."""
    return np.hypot(np.diff(ax, prepend=ax[0]), np.diff(ay, prepend=ay[0])) / px_per_mm


def _hull_perimeter(points: Array2D, px_per_mm: float) -> tuple[float, float]:
    """Convex hull perimeter and area, scaled to mm."""
    if points.shape[0] < 3:  # noqa: PLR2004
        return 0.0, 0.0
    try:
        hull = ConvexHull(points)
        return hull.area / px_per_mm, hull.volume / (px_per_mm**2)
    except Exception:
        return 0.0, 0.0


def _count_in_ranges(
    values: Array2D,
    ranges: list[tuple[float, float]],
) -> Array2D:
    """Count how many values fall in each range bracket (per frame)."""
    results = np.zeros((len(values), len(ranges)), dtype=np.float64)
    for j, (lo, hi) in enumerate(ranges):
        results[:, j] = np.sum((values >= lo) & (values <= hi), axis=1).astype(
            np.float64
        )
    return results


def _roll_median_mean(
    series: Array1D,
    window_frames: int,
) -> dict[str, Array1D]:
    """Compute rolling median and mean for a single window size."""
    return {
        "_median": (
            pl.Series(series)
            .rolling_median(window_size=window_frames, min_samples=1)
            .to_numpy()
        ),
        "_mean": (
            pl.Series(series)
            .rolling_mean(window_size=window_frames, min_samples=1)
            .to_numpy()
        ),
    }


def _ffill_bfill(arr: Array2D) -> Array2D:
    """Forward-fill then backward-fill NaN values along axis 0."""
    mask = np.isnan(arr)
    if not mask.any():
        return arr
    idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    arr = np.take_along_axis(arr, idx, axis=0)
    mask = np.isnan(arr)
    if mask.any():
        idx = np.where(~mask, np.arange(mask.shape[0])[:, None], mask.shape[0] - 1)
        np.minimum.accumulate(idx[::-1], axis=0, out=idx[::-1])
        arr = np.take_along_axis(arr, idx, axis=0)
    return arr


def _pct_rank(vals: Array1D) -> Array1D:
    """Percentile rank (replicating pandas .rank(pct=True))."""
    n = len(vals)
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    order = np.argsort(vals)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64) / n
    return ranks


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════


def extract_features(
    keypoints_df: pl.DataFrame,
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
) -> pl.DataFrame:
    """Extract generic features from preprocessed keypoints.

    Features are computed programmatically from the individuals and bodyparts
    specified in the config. No semantic bodypart roles are assumed.

    Parameters
    ----------
    keypoints_df : pl.DataFrame
        Long-form KEYPOINTS_SCHEMA DataFrame.
    config : ExperimentConfig
        Experiment configuration (extract_features.individuals, .bodyparts).
    metadata : ExperimentMetadata
        Experiment metadata (fps, px_per_mm).

    Returns:
    -------
    pl.DataFrame
        Wide features DataFrame with frame index.
    """
    cfg = config.require_extract_features()

    check_bpts_exist(keypoints_df, cfg.bodyparts)

    features_df = compute_features(
        keypoints_df.filter(
            pl.col("individual").is_in(cfg.individuals),
            pl.col("bodypart").is_in(cfg.bodyparts),
        ),
        individuals=cfg.individuals,
        bodyparts=cfg.bodyparts,
        fps=metadata.require_fps(),
        px_per_mm=metadata.require_px_per_mm(),
    )
    logger.info("Exported features to disk.")
    return features_df


# ═══════════════════════════════════════════════════════════════════════════════
# Main feature computation
# ═══════════════════════════════════════════════════════════════════════════════


def compute_features(
    keypoints_df: pl.DataFrame,
    individuals: list[str],
    bodyparts: list[str],
    fps: float,
    px_per_mm: float,
) -> pl.DataFrame:
    """Compute generic features from long-form keypoints.

    All features are programmatic — derived from the cartesian product of
    individuals and bodyparts. No semantic bodypart roles.

    Parameters
    ----------
    keypoints_df : pl.DataFrame
        Long-form KEYPOINTS_SCHEMA, pre-filtered to requested individuals+bps.
    individuals : list[str]
        Ordered individual labels.
    bodyparts : list[str]
        Ordered bodypart labels.
    fps : float
        Frames per second.
    px_per_mm : float
        Pixels per mm scale factor.

    Returns:
    -------
    pl.DataFrame
        Wide features DataFrame with frame index.
    """
    n_frames = keypoints_df.select("frame").n_unique()

    roll_windows: list[int] = sorted(
        {w for d in ROLL_WINDOW_DIVISORS if (w := max(2, int(fps / d))) <= n_frames / 2}
    )

    arrs, arr_prob = _pivot_to_wide(keypoints_df, individuals, bodyparts)

    features: dict[str, Array1D] = {}

    features |= _compute_within_distances(arrs, individuals, bodyparts, px_per_mm)
    features |= _compute_cross_distances(arrs, individuals, bodyparts, px_per_mm)
    features |= _compute_movements(arrs, individuals, bodyparts, px_per_mm)
    features |= _compute_hull(arrs, individuals, px_per_mm)
    features |= _compute_cdist_stats(arrs, individuals, px_per_mm)
    features |= _compute_total_movements(features, individuals, bodyparts)
    features |= _compute_total_cdist(features, individuals)
    features |= _compute_probability(arr_prob)
    features |= _compute_rolling(features, roll_windows)
    features |= _compute_deviations(features)
    features |= _compute_percentile_ranks(features)

    return _build_output_df(keypoints_df, features)


# ═══════════════════════════════════════════════════════════════════════════════
# Pivot: long-form Polars → wide numpy arrays
# ═══════════════════════════════════════════════════════════════════════════════


def _pivot_to_wide(
    keypoints_df: pl.DataFrame,
    individuals: list[str],
    bodyparts: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Convert long-form keypoints to wide numpy arrays.

    Parameters
    ----------
    keypoints_df : pl.DataFrame
        KEYPOINTS_SCHEMA DataFrame, pre-filtered.

    Returns:
    -------
    arrs : dict[str, np.ndarray]
        {individual_label: (n_frames, 2 * n_bodyparts)}
        Columns are interleaved [bp0_x, bp0_y, bp1_x, bp1_y, ...].
    arr_prob : np.ndarray
        (n_frames, n_individuals * n_bodyparts) — likelihoods.
    """
    unique_frames = (
        keypoints_df.select("frame").unique().sort("frame").to_series().to_numpy()
    )
    n_frames = unique_frames.shape[0]
    n_bp = len(bodyparts)

    frame_to_pos = {int(frame): pos for pos, frame in enumerate(unique_frames)}

    def _positions(frame_vals: np.ndarray) -> np.ndarray:
        return np.array([frame_to_pos[int(f)] for f in frame_vals], dtype=np.intp)

    arrs: dict[str, np.ndarray] = {}
    for indiv in individuals:
        arr = np.full((n_frames, 2 * n_bp), np.nan, dtype=np.float64)
        for bp_i, bp in enumerate(bodyparts):
            bp_data = keypoints_df.filter(
                pl.col("individual") == indiv,
                pl.col("bodypart") == bp,
            ).sort("frame")
            pos = _positions(bp_data.select("frame").to_series().to_numpy())
            x_vals = bp_data.select("x").to_series().to_numpy()
            y_vals = bp_data.select("y").to_series().to_numpy()
            arr[pos, 2 * bp_i] = x_vals
            arr[pos, 2 * bp_i + 1] = y_vals
        arrs[indiv] = _ffill_bfill(arr)

    n_prob_cols = len(individuals) * n_bp
    arr_prob = np.full((n_frames, n_prob_cols), np.nan, dtype=np.float64)
    for ind_i, indiv in enumerate(individuals):
        for bp_i, bp in enumerate(bodyparts):
            bp_data = keypoints_df.filter(
                pl.col("individual") == indiv,
                pl.col("bodypart") == bp,
            ).sort("frame")
            pos = _positions(bp_data.select("frame").to_series().to_numpy())
            p_vals = bp_data.select("likelihood").to_series().to_numpy()
            prob_col = ind_i * n_bp + bp_i
            arr_prob[pos, prob_col] = p_vals
    arr_prob = _ffill_bfill(arr_prob)

    return arrs, arr_prob


def _get_xy(arr: Array2D, bp_i: int) -> tuple[Array1D, Array1D]:
    """Extract (x, y) columns for bodypart at index bp_i."""
    return arr[:, 2 * bp_i], arr[:, 2 * bp_i + 1]


# ═══════════════════════════════════════════════════════════════════════════════
# Feature groups
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_within_distances(
    arrs: dict[str, Array2D],
    individuals: list[str],
    bodyparts: list[str],
    px_per_mm: float,
) -> dict[str, Array1D]:
    """All pairwise distances between bodyparts within each individual.

    Column: ``{indiv}_{bp_a}_to_{bp_b}_dist``.
    """
    f: dict[str, Array1D] = {}
    for indiv in individuals:
        arr = arrs[indiv]
        for (i, bp_a), (j, bp_b) in combinations(enumerate(bodyparts), 2):
            ax, ay = _get_xy(arr, i)
            bx, by = _get_xy(arr, j)
            f[f"{indiv}_{bp_a}_to_{bp_b}_dist"] = _euclidean(ax, ay, bx, by, px_per_mm)
    return f


def _compute_cross_distances(
    arrs: dict[str, Array2D],
    individuals: list[str],
    bodyparts: list[str],
    px_per_mm: float,
) -> dict[str, Array1D]:
    """All pairwise distances between bodyparts across different individuals.

    Column: ``{indiv_a}_{bp_a}_to_{indiv_b}_{bp_b}_dist``.
    Skipped if fewer than 2 individuals.
    """
    f: dict[str, Array1D] = {}
    n = len(individuals)
    if n < 2:  # noqa: PLR2004
        return f
    for (_i, ind_a), (_j, ind_b) in combinations(enumerate(individuals), 2):
        arr_a, arr_b = arrs[ind_a], arrs[ind_b]
        for bp_i, bp_a in enumerate(bodyparts):
            ax, ay = _get_xy(arr_a, bp_i)
            for bp_j, bp_b in enumerate(bodyparts):
                bx, by = _get_xy(arr_b, bp_j)
                f[f"{ind_a}_{bp_a}_to_{ind_b}_{bp_b}_dist"] = _euclidean(
                    ax,
                    ay,
                    bx,
                    by,
                    px_per_mm,
                )
    return f


def _compute_movements(
    arrs: dict[str, Array2D],
    individuals: list[str],
    bodyparts: list[str],
    px_per_mm: float,
) -> dict[str, Array1D]:
    """Per-bodypart frame-to-frame movement for each individual.

    Column: ``{indiv}_{bp}_movement``.
    """
    f: dict[str, Array1D] = {}
    for indiv in individuals:
        arr = arrs[indiv]
        for bp_i, bp in enumerate(bodyparts):
            ax, ay = _get_xy(arr, bp_i)
            f[f"{indiv}_{bp}_movement"] = _movement_frame_to_frame(ax, ay, px_per_mm)
    return f


def _compute_hull(
    arrs: dict[str, Array2D],
    individuals: list[str],
    px_per_mm: float,
) -> dict[str, Array1D]:
    """Convex hull perimeter and areal change per individual.

    Columns: ``{indiv}_hull_perimeter``, ``{indiv}_hull_area_change``.
    """
    f: dict[str, Array1D] = {}
    for indiv in individuals:
        arr = arrs[indiv]
        n_frames = arr.shape[0]
        perimeters = np.zeros(n_frames, dtype=np.float64)
        areas = np.zeros(n_frames, dtype=np.float64)
        for frame_i in range(n_frames):
            points = arr[frame_i].reshape(-1, 2)
            valid = ~np.isnan(points).any(axis=1)
            if valid.sum() >= 3:  # noqa: PLR2004
                perimeters[frame_i], areas[frame_i] = _hull_perimeter(
                    points[valid],
                    px_per_mm,
                )
        f[f"{indiv}_hull_perimeter"] = perimeters

        areas_shifted = np.empty_like(areas)
        areas_shifted[0] = areas[0]
        areas_shifted[1:] = areas[:-1]
        f[f"{indiv}_hull_area_change"] = areas_shifted - areas
    return f


def _compute_cdist_stats(
    arrs: dict[str, Array2D],
    individuals: list[str],
    px_per_mm: float,
) -> dict[str, Array1D]:
    """Cdist statistics (max, min, mean, sum) of within-hull distances.

    Columns: ``{indiv}_cdist_max``, _min, _mean, _sum.
    """
    f: dict[str, Array1D] = {}
    for indiv in individuals:
        arr = arrs[indiv]
        n_frames = arr.shape[0]
        large = np.zeros(n_frames, dtype=np.float64)
        small = np.zeros(n_frames, dtype=np.float64)
        mean_ = np.zeros(n_frames, dtype=np.float64)
        sum_ = np.zeros(n_frames, dtype=np.float64)
        for frame_i in range(n_frames):
            points = arr[frame_i].reshape(-1, 2)
            valid = ~np.isnan(points).any(axis=1)
            pts = points[valid]
            if len(pts) >= 2:  # noqa: PLR2004
                dists = pdist(pts)
                if len(dists) > 0:
                    dists_mm = dists / px_per_mm
                    large[frame_i] = np.max(dists_mm)
                    small[frame_i] = np.min(dists_mm)
                    mean_[frame_i] = np.mean(dists_mm)
                    sum_[frame_i] = np.sum(dists_mm)
        f[f"{indiv}_cdist_max"] = large
        f[f"{indiv}_cdist_min"] = small
        f[f"{indiv}_cdist_mean"] = mean_
        f[f"{indiv}_cdist_sum"] = sum_
    return f


def _compute_total_movements(
    features: dict[str, Array1D],
    individuals: list[str],
    bodyparts: list[str],
) -> dict[str, Array1D]:
    """Sum of all bodypart movements per individual and grand total.

    Columns: ``total_movement_{indiv}``, ``total_movement_all``.
    """
    f: dict[str, Array1D] = {}
    for indiv in individuals:
        keys = [f"{indiv}_{bp}_movement" for bp in bodyparts]
        existing = [k for k in keys if k in features]
        if existing:
            f[f"total_movement_{indiv}"] = sum(features[k] for k in existing)

    total_keys = [f"total_movement_{indiv}" for indiv in individuals]
    existing_total = [k for k in total_keys if k in f]
    if existing_total:
        f["total_movement_all"] = sum(f[k] for k in existing_total)
    return f


def _compute_total_cdist(
    features: dict[str, Array1D],
    individuals: list[str],
) -> dict[str, Array1D]:
    """Sum of per-individual cdist sums across all individuals.

    Column: ``cdist_sum_all``.
    """
    keys = [f"{indiv}_cdist_sum" for indiv in individuals]
    existing = [k for k in keys if k in features]
    if existing:
        return {"cdist_sum_all": sum(features[k] for k in existing)}
    return {}


def _compute_probability(arr_prob: Array2D) -> dict[str, Array1D]:
    """Likelihood-based probability features.

    Columns: ``sum_probabilities``, ``low_prob_detections_0.1/0.5/0.75``.
    """
    counts = _count_in_ranges(
        arr_prob,
        [(0.0, 0.1), (0.0, 0.5), (0.0, 0.75)],
    )
    return {
        "sum_probabilities": np.sum(arr_prob, axis=1),
        "low_prob_detections_0.1": counts[:, 0],
        "low_prob_detections_0.5": counts[:, 1],
        "low_prob_detections_0.75": counts[:, 2],
    }


def _compute_rolling(
    features: dict[str, Array1D],
    roll_windows: list[int],
) -> dict[str, Array1D]:
    """Rolling window median/mean for all base numeric features.

    Columns: ``{base_name}_median/mean_w{frames}``.

    Only applies to base (non-derived) features. Windows are distinct integer
    frame counts.
    """
    aggs: dict[str, Array1D] = {}

    exclude = {
        "low_prob_detections",
        "sum_probabilities",
    }
    base_keys = [
        k
        for k in features
        if (
            not any(k.startswith(p) for p in exclude)
            and "_median_" not in k
            and "_mean_" not in k
            and "_deviation" not in k
            and "_percentile_rank" not in k
        )
    ]

    for wf in roll_windows:
        for key in base_keys:
            stats = _roll_median_mean(features[key], wf)
            aggs[f"{key}_median_w{wf}"] = stats["_median"]
            aggs[f"{key}_mean_w{wf}"] = stats["_mean"]

    return aggs


def _compute_deviations(
    features: dict[str, Array1D],
) -> dict[str, Array1D]:
    """Deviation features: mean(feature) - current(feature).

    Columns: ``{name}_deviation``.

    Computed for aggregate features (totals, cdist, hull) and rolling means.
    """
    aggs: dict[str, Array1D] = {}

    dev_prefixes = (
        "total_movement",
        "cdist_sum",
        "sum_probabilities",
    )
    dev_suffixes = (
        "_hull_perimeter",
        "_cdist_mean",
        "_cdist_max",
        "_cdist_min",
    )
    for key, val in features.items():
        if "_deviation" in key or "_percentile_rank" in key:
            continue
        if key.startswith(dev_prefixes) or key.endswith(dev_suffixes):
            aggs[f"{key}_deviation"] = val.mean() - val

    # Also compute deviations for rolling mean features
    for key in list(features.keys()):
        if "_mean_" in key and "_deviation" not in key and "percentile" not in key:
            aggs[f"{key}_deviation"] = features[key].mean() - features[key]

    return aggs


def _compute_percentile_ranks(
    features: dict[str, Array1D],
) -> dict[str, Array1D]:
    """Percentile rank features.

    Columns: ``{name}_percentile_rank``.

    Computed for total movements, cdist sums, and their deviations only.
    """
    aggs: dict[str, Array1D] = {}

    rank_prefixes = ("total_movement", "cdist_sum")
    for key, val in features.items():
        if "_percentile_rank" in key:
            continue
        if key.startswith(rank_prefixes):
            aggs[f"{key}_percentile_rank"] = _pct_rank(val)

    return aggs


def _build_output_df(
    keypoints_df: pl.DataFrame,
    features: dict[str, Array1D],
) -> pl.DataFrame:
    """Build final Polars DataFrame with frame index + all feature columns."""
    frames = keypoints_df.select("frame").unique().sort("frame").to_series().to_numpy()
    col_data: dict[str, np.ndarray] = {"frame": frames.flatten()}
    col_data |= features
    return pl.DataFrame(col_data).with_columns(pl.col("frame").cast(pl.Int64))
