"""Feature extraction from preprocessed keypoints using SimBA feature math.

Replicates the SimBA ExtractFeaturesFrom16bps pipeline natively in
Polars + NumPy + SciPy.
No separate conda environment or subprocess required.

Source of truth:
https://github.com/sgoldenlab/simba/blob/master/simba/feature_extractors/feature_extractor_16bp.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from loguru import logger
from scipy.spatial import ConvexHull

from behavysis.constants.bodypoints import BPMAP_SIMBA, INDIVS_SIMBA
from behavysis.transforms.keypoint import check_bpts_exist

if TYPE_CHECKING:
    from behavysis.models import ExperimentConfig, ExperimentMetadata


def extract_features(
    keypoints_df: pl.DataFrame,
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
) -> pl.DataFrame:
    """Extract SimBA-compatible features from preprocessed keypoints.

    Parameters
    ----------
    keypoints_df : pl.DataFrame
        Long-form KEYPOINTS_SCHEMA DataFrame.
    config : ExperimentConfig
        Experiment configuration.
    metadata : ExperimentMetadata
        Experiment metadata (fps, px_per_mm, etc.).

    Returns:
    -------
    pl.DataFrame
        Wide features DataFrame with frame index and SimBA-compatible columns.
    """
    cfg = config.require_extract_features()

    check_bpts_exist(keypoints_df, cfg.bodyparts)

    features_df = compute_simba_features(
        keypoints_df.filter(
            pl.col("individual").is_in(cfg.individuals),
            pl.col("bodypart").is_in(cfg.bodyparts),
        ),
        fps=metadata.require_fps(),
        px_per_mm=metadata.require_px_per_mm(),
    )
    logger.info("Exported SimBA features to disk.")
    return features_df


# ═══════════════════════════════════════════════════════════════════════════════
# Body-part naming conventions
# SimBA source:
#   bp_names.csv — Ear_left_1, Ear_right_1, Nose_1, Center_1, Lat_left_1, …
#   feature_extractor_16bp.py — hard-codes movement suffixes that differ from
#   the bp_names.csv names (e.g. "centroid" not "center", "left_ear" not "ear_left")
# ═══════════════════════════════════════════════════════════════════════════════

BP_XY_IDX: dict[str, tuple[int, int]] = {
    "Ear_left": (0, 1),
    "Ear_right": (2, 3),
    "Nose": (4, 5),
    "Center": (6, 7),
    "Lat_left": (8, 9),
    "Lat_right": (10, 11),
    "Tail_base": (12, 13),
    "Tail_end": (14, 15),
}

MOVEMENT_BP_NAMES: dict[str, str] = {
    "Ear_left": "left_ear",
    "Ear_right": "right_ear",
    "Nose": "nose",
    "Center": "centroid",
    "Lat_left": "lateral_left",
    "Lat_right": "lateral_right",
    "Tail_base": "tail_base",
    "Tail_end": "tail_end",
}

SIMBA_BODY_PARTS: list[str] = list(BP_XY_IDX.keys())

# SimBA Options.ROLLING_WINDOW_DIVISORS subset used for feature extraction
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
# Typing aliases
# ═══════════════════════════════════════════════════════════════════════════════

type Array1D = np.ndarray[tuple[int], np.dtype[np.float64]]
type Array2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]

# ═══════════════════════════════════════════════════════════════════════════════
# Vectorized math helpers
# Replicating SimBA's numba-jitted functions in pure NumPy.
# ═══════════════════════════════════════════════════════════════════════════════


def _get_xy(arr: Array2D, bp_name: str) -> tuple[Array1D, Array1D]:
    """Extract (x_array, y_array) for a body-part from the wide array."""
    ix, iy = BP_XY_IDX[bp_name]
    return arr[:, ix], arr[:, iy]


def _euclidean(
    ax: Array1D,
    ay: Array1D,
    bx: Array1D,
    by: Array1D,
    px_per_mm: float,
) -> Array1D:
    """Vectorized Euclidean distance between two point sets, scaled to mm."""
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2) / px_per_mm


def _angle3pt_vectorized(
    nose_x: Array1D,
    nose_y: Array1D,
    center_x: Array1D,
    center_y: Array1D,
    tail_x: Array1D,
    tail_y: Array1D,
) -> Array1D:
    """Replicates SimBA angle3pt_vectorized: 3-point angle at center.

    SimBA computes:
        degrees(
            atan2(|tail_x-cx|, |tail_y-cy|)
            - atan2(|nose_x-cx|, |nose_y-cy|)
        )
    """
    return np.degrees(
        np.abs(
            np.arctan2(np.abs(tail_x - center_x), np.abs(tail_y - center_y))
            - np.arctan2(np.abs(nose_x - center_x), np.abs(nose_y - center_y)),
        ),
    )


def _movement_bp(ax: Array1D, ay: Array1D, px_per_mm: float) -> Array1D:
    """Frame-to-frame movement for a single body-part.

    SimBA: shifts by 1, fills first with original, then euclidean distance.
    """
    ax_shifted = np.empty_like(ax)
    ay_shifted = np.empty_like(ay)
    ax_shifted[0] = ax[0]
    ay_shifted[0] = ay[0]
    ax_shifted[1:] = ax[:-1]
    ay_shifted[1:] = ay[:-1]
    return _euclidean(ax_shifted, ay_shifted, ax, ay, px_per_mm)


def _hull_perimeter(points: Array2D, px_per_mm: float) -> tuple[float, float]:
    """Convex hull perimeter for a set of 2D points, scaled to mm.

    Replaces SimBA's numba jitted_hull. Uses scipy.spatial.ConvexHull.
    Returns (perimeter_mm, area_mm2).
    """
    if points.shape[0] < 3:  # noqa: PLR2004
        return 0.0, 0.0
    try:
        hull = ConvexHull(points)
        return hull.area / px_per_mm, hull.volume / (px_per_mm**2)
    except Exception:
        return 0.0, 0.0


def _cdist(points: Array2D) -> Array2D:
    """Pairwise Euclidean distances between all points (like scipy cdist)."""
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def _count_in_ranges(
    values: Array2D,
    ranges: list[tuple[float, float]],
) -> Array2D:
    """Count how many values fall in each range bracket (per frame).

    values: shape (n_frames, n_bodyparts) array of probabilities.
    ranges: list of (low, high) tuples.
    Returns: shape (n_frames, n_ranges).
    """
    results = np.zeros((len(values), len(ranges)), dtype=np.float64)
    for j, (lo, hi) in enumerate(ranges):
        results[:, j] = np.sum((values >= lo) & (values <= hi), axis=1).astype(
            np.float64,
        )
    return results


def _tortuosity(
    centroid_x: Array1D,
    centroid_y: Array1D,
    window_frames: int,
) -> Array1D:
    """Path tortuosity: 3-point sliding window curvature sum / (2*pi).

    Replicates SimBA's strided approach.
    """
    n = len(centroid_x)
    if n < 3 or window_frames < 2:  # noqa: PLR2004
        return np.zeros(n, dtype=np.float64)

    window_samples = min(max(int(window_frames), 3), n)

    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window_samples + 1)
        end = i + 1
        seg_x = centroid_x[start:end]
        seg_y = centroid_y[start:end]
        m = len(seg_x)
        if m < 3:
            continue
        curvature = 0.0
        for j in range(m - 2):
            ax, ay = seg_x[j], seg_y[j]
            bx, by = seg_x[j + 1], seg_y[j + 1]
            cx, cy = seg_x[j + 2], seg_y[j + 2]
            ang = np.degrees(
                np.abs(
                    np.arctan2(np.abs(cx - bx), np.abs(cy - by))
                    - np.arctan2(np.abs(ax - bx), np.abs(ay - by)),
                ),
            )
            curvature += ang
        result[i] = curvature / (2 * np.pi)
    return result


def _roll_median_mean_sum(
    series: Array1D,
    window_frames: int,
) -> dict[str, Array1D]:
    """Compute rolling median, mean, sum for a single window size."""
    return {
        "_median": pl.Series(series)
        .rolling_median(
            window_size=window_frames,
            min_samples=1,
        )
        .to_numpy(),
        "_mean": pl.Series(series)
        .rolling_mean(
            window_size=window_frames,
            min_samples=1,
        )
        .to_numpy(),
        "_sum": pl.Series(series)
        .rolling_sum(
            window_size=window_frames,
            min_samples=1,
        )
        .to_numpy(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main feature extraction
# ═══════════════════════════════════════════════════════════════════════════════


def compute_simba_features(
    keypoints_df: pl.DataFrame,
    fps: float,
    px_per_mm: float,
) -> pl.DataFrame:
    """Compute SimBA features from Polars long-form keypoints.

    Column names match SimBA ExtractFeaturesFrom16bps output exactly.

    Parameters
    ----------
    keypoints_df : pl.DataFrame
        Long-form KEYPOINTS_SCHEMA DataFrame, pre-filtered to 2 animals x 8 bps.
    fps : float
        Frames per second.
    px_per_mm : float
        Pixels per mm scale factor.

    Returns:
    -------
    pl.DataFrame
        Wide features DataFrame with frame index and SimBA-compatible columns.
    """
    n_frames = keypoints_df.select("frame").n_unique()

    roll_windows: list[float] = []
    for d in ROLL_WINDOW_DIVISORS:
        w = max(2, int(fps / d))
        if w <= n_frames / 2:
            roll_windows.append(d)

    arr_m1, arr_m2, arr_prob = _pivot_to_wide(keypoints_df)

    features: dict[str, Array1D] = {}

    features |= _compute_distances(arr_m1, arr_m2, px_per_mm)
    features |= _compute_movements(arr_m1, arr_m2, px_per_mm)
    features |= _compute_hull(arr_m1, arr_m2, px_per_mm)
    features |= _compute_cdist(arr_m1, arr_m2, px_per_mm)
    features |= _compute_angles(arr_m1, arr_m2)
    features |= _compute_aggregates(features)
    features |= _compute_tail_end_relative_raw(features)
    features |= _compute_probability(arr_prob)
    features |= _compute_rolling(features, roll_windows, fps)
    features |= _compute_deviations(features, roll_windows, fps)
    features |= _compute_tortuosities(arr_m1, arr_m2, roll_windows, fps)
    features |= _compute_percentile_ranks(features)

    return _build_output_df(keypoints_df, features)


# ═══════════════════════════════════════════════════════════════════════════════
# Pivot: long-form Polars → wide numpy arrays
# ═══════════════════════════════════════════════════════════════════════════════


def _pivot_to_wide(
    keypoints_df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert long-form keypoints to wide numpy arrays per animal.

    Returns (arr_m1, arr_m2, arr_prob):
        arr_m1  : (n_frames, 16) — x,y for 8 body-parts of animal 1
        arr_m2  : (n_frames, 16) — x,y for 8 body-parts of animal 2
        arr_prob: (n_frames, 16) — likelihoods for both animals (8 bp x 2)
    """
    n_frames = keypoints_df.select("frame").n_unique()

    arr_m1 = np.full((n_frames, 16), np.nan, dtype=np.float64)
    arr_m2 = np.full((n_frames, 16), np.nan, dtype=np.float64)
    arr_prob = np.full((n_frames, 16), np.nan, dtype=np.float64)

    for indiv in INDIVS_SIMBA:
        is_m1 = indiv == "mouse1marked"
        arr = arr_m1 if is_m1 else arr_m2
        prob_offset = 0 if is_m1 else 8

        for our_bp, simba_bp in BPMAP_SIMBA.items():
            if simba_bp not in BP_XY_IDX:
                continue
            ix, iy = BP_XY_IDX[simba_bp]
            bp_data = keypoints_df.filter(
                pl.col("individual") == indiv,
                pl.col("bodypart") == our_bp,
            ).sort("frame")

            frames = bp_data.select("frame").to_series().to_numpy()
            x_vals = bp_data.select("x").to_series().to_numpy()
            y_vals = bp_data.select("y").to_series().to_numpy()
            p_vals = bp_data.select("likelihood").to_series().to_numpy()

            arr[frames, ix] = x_vals
            arr[frames, iy] = y_vals

            prob_idx = prob_offset + SIMBA_BODY_PARTS.index(simba_bp)
            arr_prob[frames, prob_idx] = p_vals

    arr_m1 = _ffill_bfill(arr_m1)
    arr_m2 = _ffill_bfill(arr_m2)
    arr_prob = _ffill_bfill(arr_prob)

    return arr_m1, arr_m2, arr_prob


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


# ═══════════════════════════════════════════════════════════════════════════════
# Feature groups
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_distances(
    arr_m1: Array2D,
    arr_m2: Array2D,
    px_per_mm: float,
) -> dict[str, Array1D]:
    """Compute all inter-body-part Euclidean distance features.

    Matches the order and names from SimBA's run() method exactly.
    """

    def _d(a: Array2D, b: Array2D, bp_a: str, bp_b: str) -> Array1D:
        ax, ay = _get_xy(a, bp_a)
        bx, by = _get_xy(b, bp_b)
        return _euclidean(ax, ay, bx, by, px_per_mm)

    f: dict[str, Array1D] = {}

    # Animal 1
    f["Mouse_1_nose_to_tail"] = _d(arr_m1, arr_m1, "Nose", "Tail_base")
    f["Mouse_1_width"] = _d(arr_m1, arr_m1, "Lat_left", "Lat_right")
    f["Mouse_1_Ear_distance"] = _d(arr_m1, arr_m1, "Ear_left", "Ear_right")
    f["Mouse_1_Nose_to_centroid"] = _d(arr_m1, arr_m1, "Nose", "Center")
    f["Mouse_1_Nose_to_lateral_left"] = _d(arr_m1, arr_m1, "Nose", "Lat_left")
    f["Mouse_1_Nose_to_lateral_right"] = _d(arr_m1, arr_m1, "Nose", "Lat_right")
    f["Mouse_1_Centroid_to_lateral_left"] = _d(arr_m1, arr_m1, "Center", "Lat_left")
    f["Mouse_1_Centroid_to_lateral_right"] = _d(arr_m1, arr_m1, "Center", "Lat_right")

    # Animal 2
    f["Mouse_2_nose_to_tail"] = _d(arr_m2, arr_m2, "Nose", "Tail_base")
    f["Mouse_2_width"] = _d(arr_m2, arr_m2, "Lat_left", "Lat_right")
    f["Mouse_2_Ear_distance"] = _d(arr_m2, arr_m2, "Ear_left", "Ear_right")
    f["Mouse_2_Nose_to_centroid"] = _d(arr_m2, arr_m2, "Nose", "Center")
    f["Mouse_2_Nose_to_lateral_left"] = _d(arr_m2, arr_m2, "Nose", "Lat_left")
    f["Mouse_2_Nose_to_lateral_right"] = _d(arr_m2, arr_m2, "Nose", "Lat_right")
    f["Mouse_2_Centroid_to_lateral_left"] = _d(arr_m2, arr_m2, "Center", "Lat_left")
    f["Mouse_2_Centroid_to_lateral_right"] = _d(arr_m2, arr_m2, "Center", "Lat_right")

    # Cross-animal
    f["Centroid_distance"] = _d(arr_m2, arr_m1, "Center", "Center")
    f["Nose_to_nose_distance"] = _d(arr_m2, arr_m1, "Nose", "Nose")
    f["M1_Nose_to_M2_lat_left"] = _d(arr_m1, arr_m2, "Nose", "Lat_left")
    f["M1_Nose_to_M2_lat_right"] = _d(arr_m1, arr_m2, "Nose", "Lat_right")
    f["M2_Nose_to_M1_lat_left"] = _d(arr_m2, arr_m1, "Nose", "Lat_left")
    f["M2_Nose_to_M1_lat_right"] = _d(arr_m2, arr_m1, "Nose", "Lat_right")
    f["M1_Nose_to_M2_tail_base"] = _d(arr_m1, arr_m2, "Nose", "Tail_base")
    f["M2_Nose_to_M1_tail_base"] = _d(arr_m2, arr_m1, "Nose", "Tail_base")

    return f


def _compute_movements(
    arr_m1: Array2D,
    arr_m2: Array2D,
    px_per_mm: float,
) -> dict[str, Array1D]:
    """Frame-to-frame movement for each body-part.

    SimBA hard-codes movement suffixes that don't match bp_names.csv lowercase.
    e.g. Center → "centroid", Ear_left → "left_ear", Lat_left → "lateral_left".
    """
    f: dict[str, Array1D] = {}
    for bp_simba, bp_movement in MOVEMENT_BP_NAMES.items():
        ix, iy = BP_XY_IDX[bp_simba]
        f[f"Movement_mouse_1_{bp_movement}"] = _movement_bp(
            arr_m1[:, ix],
            arr_m1[:, iy],
            px_per_mm,
        )
        f[f"Movement_mouse_2_{bp_movement}"] = _movement_bp(
            arr_m2[:, ix],
            arr_m2[:, iy],
            px_per_mm,
        )
    return f


def _compute_hull(
    arr_m1: Array2D,
    arr_m2: Array2D,
    px_per_mm: float,
) -> dict[str, Array1D]:
    """Convex hull perimeter and polygon size change features.

    SimBA uses Mouse_1, Mouse_2 (not Mouse_M1/Mouse_M2).
    """
    return {**_hull_one(arr_m1, px_per_mm, "1"), **_hull_one(arr_m2, px_per_mm, "2")}


def _hull_one(arr: Array2D, px_per_mm: float, label: str) -> dict[str, Array1D]:
    """SimBA: jitted_hull perimeter + shifted-current polygon size change."""
    n = arr.shape[0]
    perimeters = np.zeros(n, dtype=np.float64)
    areas = np.zeros(n, dtype=np.float64)

    for i in range(n):
        points = arr[i].reshape(-1, 2)
        valid = ~np.isnan(points).any(axis=1)
        if valid.sum() >= 3:  # noqa: PLR2004
            perimeters[i], areas[i] = _hull_perimeter(points[valid], px_per_mm)

    # SimBA Mouse_1_poly_area_shifted - Mouse_1_poly_area  (signed)
    areas_shifted = np.empty_like(areas)
    areas_shifted[0] = areas[0]
    areas_shifted[1:] = areas[:-1]
    size_change = areas_shifted - areas

    ml = label.lower()
    return {
        f"Mouse_{ml}_poly_area": perimeters,
        f"Mouse_{ml}_polygon_size_change": size_change,
    }


def _compute_cdist(
    arr_m1: Array2D,
    arr_m2: Array2D,
    px_per_mm: float,
) -> dict[str, Array1D]:
    """Pairwise distance stats (cdist) within each animal's hull."""
    f: dict[str, Array1D] = {}
    f |= _cdist_one(arr_m1, px_per_mm, "M1")
    f |= _cdist_one(arr_m2, px_per_mm, "M2")
    f["Sum_euclidean_distance_hull_M1_M2"] = (
        f["M1_hull_sum_euclidean"] + f["M2_hull_sum_euclidean"]
    )
    return f


def _cdist_one(arr: Array2D, px_per_mm: float, label: str) -> dict[str, Array1D]:
    """SimBA: hull_large/small/mean/sum_euclidean via cdist per frame."""
    n = arr.shape[0]
    large = np.zeros(n, dtype=np.float64)
    small = np.zeros(n, dtype=np.float64)
    mean_ = np.zeros(n, dtype=np.float64)
    sum_ = np.zeros(n, dtype=np.float64)

    for i in range(n):
        points = arr[i].reshape(-1, 2)
        valid = ~np.isnan(points).any(axis=1)
        pts = points[valid]
        if len(pts) >= 2:  # noqa: PLR2004
            dists = _cdist(pts)
            triu = dists[np.triu_indices_from(dists, k=1)]
            if len(triu) > 0:
                triu_mm = triu / px_per_mm
                large[i] = np.max(triu_mm)
                small[i] = np.min(triu_mm)
                mean_[i] = np.mean(triu_mm)
                sum_[i] = np.sum(triu_mm)

    return {
        f"{label}_hull_large_euclidean": large,
        f"{label}_hull_small_euclidean": small,
        f"{label}_hull_mean_euclidean": mean_,
        f"{label}_hull_sum_euclidean": sum_,
    }


def _compute_angles(arr_m1: Array2D, arr_m2: Array2D) -> dict[str, Array1D]:
    """3-point angle (nose→center→tail_base) per animal."""
    n1x, n1y = _get_xy(arr_m1, "Nose")
    c1x, c1y = _get_xy(arr_m1, "Center")
    t1x, t1y = _get_xy(arr_m1, "Tail_base")
    m1_angle = _angle3pt_vectorized(n1x, n1y, c1x, c1y, t1x, t1y)

    n2x, n2y = _get_xy(arr_m2, "Nose")
    c2x, c2y = _get_xy(arr_m2, "Center")
    t2x, t2y = _get_xy(arr_m2, "Tail_base")
    m2_angle = _angle3pt_vectorized(n2x, n2y, c2x, c2y, t2x, t2y)

    return {
        "Mouse_1_angle": m1_angle,
        "Mouse_2_angle": m2_angle,
        "Total_angle_both_mice": m1_angle + m2_angle,
    }


def _compute_aggregates(
    features: dict[str, Array1D],
) -> dict[str, Array1D]:
    """Sum aggregate movement features.

    SimBA: Total_movement_all_bodyparts_M1 sums 7 body-parts, EXCLUDES centroid.
    """
    aggs: dict[str, Array1D] = {}
    aggs["Total_movement_centroids"] = (
        features["Movement_mouse_1_centroid"] + features["Movement_mouse_2_centroid"]
    )
    aggs["Total_movement_tail_ends"] = (
        features["Movement_mouse_1_tail_end"] + features["Movement_mouse_2_tail_end"]
    )

    # 7 body-parts (EXCLUDES centroid)
    total_bps = [
        "nose",
        "tail_end",
        "tail_base",
        "left_ear",
        "right_ear",
        "lateral_left",
        "lateral_right",
    ]
    m1_total = np.ndarray(sum(features[f"Movement_mouse_1_{bp}"] for bp in total_bps))
    m2_total = np.ndarray(sum(features[f"Movement_mouse_2_{bp}"] for bp in total_bps))
    aggs["Total_movement_all_bodyparts_M1"] = m1_total
    aggs["Total_movement_all_bodyparts_M2"] = m2_total
    aggs["Total_movement_all_bodyparts_both_mice"] = m1_total + m2_total

    return aggs


def _compute_tail_end_relative_raw(
    features: dict[str, Array1D],
) -> dict[str, Array1D]:
    """SimBA: M1 tail_end - (tail_base + centroid + nose).

    Only computed for M1 (per SimBA source), not M2.
    """
    return {
        "Tail_end_relative_to_tail_base_centroid_nose": (
            features["Movement_mouse_1_tail_end"]
            - features["Movement_mouse_1_tail_base"]
            - features["Movement_mouse_1_centroid"]
            - features["Movement_mouse_1_nose"]
        ),
    }


def _compute_probability(arr_prob: Array2D) -> dict[str, Array1D]:
    """Probability-based features: sum of likelihoods, low-prob detection counts."""
    counts = _count_in_ranges(
        arr_prob,
        [(0.0, 0.1), (0.0, 0.5), (0.0, 0.75)],
    )
    return {
        "Sum_probabilities": np.sum(arr_prob, axis=1),
        "Low_prob_detections_0.1": counts[:, 0],
        "Low_prob_detections_0.5": counts[:, 1],
        "Low_prob_detections_0.75": counts[:, 2],
    }


def _compute_rolling(
    features: dict[str, Array1D],
    roll_windows: list[float],
    fps: float,
) -> dict[str, Array1D]:
    """Compute rolling window median/mean/sum with exact SimBA column names.

    SimBA uses raw float strings as window labels: "2.0", "5.0", "7.5", etc.
    NOT integer labels like "2", "5".
    """
    aggs: dict[str, Array1D] = {}

    for w in roll_windows:
        wf = max(2, int(fps / w))
        wl = str(w)  # SimBA: raw float string

        aggs |= _rolling_for(
            features,
            "Sum_euclidean_distance_hull_M1_M2",
            "Sum_euclid_distances_hull",
            wf,
            wl,
        )
        aggs |= _rolling_for(features, "Total_movement_centroids", "Movement", wf, wl)
        aggs |= _rolling_for(features, "Centroid_distance", "Distance", wf, wl)
        aggs |= _rolling_for(features, "Mouse_1_width", "Mouse1_width", wf, wl)
        aggs |= _rolling_for(features, "Mouse_2_width", "Mouse2_width", wf, wl)
        aggs |= _rolling_for(
            features,
            "M1_hull_mean_euclidean",
            "Mouse1_mean_euclid_distances",
            wf,
            wl,
        )
        aggs |= _rolling_for(
            features,
            "M2_hull_mean_euclidean",
            "Mouse2_mean_euclid_distances",
            wf,
            wl,
        )
        aggs |= _rolling_for(
            features,
            "M1_hull_small_euclidean",
            "Mouse1_smallest_euclid_distances",
            wf,
            wl,
        )
        aggs |= _rolling_for(
            features,
            "M2_hull_small_euclidean",
            "Mouse2_smallest_euclid_distances",
            wf,
            wl,
        )
        aggs |= _rolling_for(
            features,
            "M1_hull_large_euclidean",
            "Mouse1_largest_euclid_distances",
            wf,
            wl,
        )
        aggs |= _rolling_for(
            features,
            "M2_hull_large_euclidean",
            "Mouse2_largest_euclid_distances",
            wf,
            wl,
        )
        aggs |= _rolling_for(
            features,
            "Total_movement_all_bodyparts_both_mice",
            "Total_movement_all_bodyparts_both_mice",
            wf,
            wl,
        )
        aggs |= _rolling_for(
            features,
            "Total_movement_centroids",
            "Total_movement_centroids",
            wf,
            wl,
        )
        aggs |= _rolling_for(
            features,
            "Movement_mouse_1_tail_base",
            "Tail_base_movement_M1",
            wf,
            wl,
        )
        aggs |= _rolling_for(
            features,
            "Movement_mouse_2_tail_base",
            "Tail_base_movement_M2",
            wf,
            wl,
        )
        aggs |= _rolling_for(
            features,
            "Movement_mouse_1_centroid",
            "Centroid_movement_M1",
            wf,
            wl,
        )
        aggs |= _rolling_for(
            features,
            "Movement_mouse_2_centroid",
            "Centroid_movement_M2",
            wf,
            wl,
        )
        aggs |= _rolling_for(
            features,
            "Movement_mouse_1_tail_end",
            "Tail_end_movement_M1",
            wf,
            wl,
        )
        aggs |= _rolling_for(
            features,
            "Movement_mouse_2_tail_end",
            "Tail_end_movement_M2",
            wf,
            wl,
        )
        aggs |= _rolling_for(
            features,
            "Movement_mouse_1_nose",
            "Nose_movement_M1",
            wf,
            wl,
        )
        aggs |= _rolling_for(
            features,
            "Movement_mouse_2_nose",
            "Nose_movement_M2",
            wf,
            wl,
        )
        aggs |= _rolling_for(
            features,
            "Total_angle_both_mice",
            "Total_angle_both_mice",
            wf,
            wl,
        )

        # Tail_end_relative: computed from already-rolled mean features
        aggs |= _tail_end_relative_rolled(aggs, wl)

    return aggs


def _rolling_for(
    features: dict[str, Array1D],
    src_key: str,
    prefix: str,
    window_frames: int,
    window_label: str,
) -> dict[str, Array1D]:
    """SimBA: {prefix}_median_{label}, _mean_{label}, _sum_{label}."""
    if src_key not in features:
        return {}
    stats = _roll_median_mean_sum(features[src_key], window_frames)
    return {
        f"{prefix}_median_{window_label}": stats["_median"],
        f"{prefix}_mean_{window_label}": stats["_mean"],
        f"{prefix}_sum_{window_label}": stats["_sum"],
    }


def _tail_end_relative_rolled(
    features: dict[str, Array1D],
    window_label: str,
) -> dict[str, Array1D]:
    """SimBA: (rolled_tail_end - rolled_tail_base - rolled_centroid - rolled_nose)."""
    aggs: dict[str, Array1D] = {}
    for m in ["M1", "M2"]:
        te = f"Tail_end_movement_{m}_mean_{window_label}"
        tb = f"Tail_base_movement_{m}_mean_{window_label}"
        cm = f"Centroid_movement_{m}_mean_{window_label}"
        ns = f"Nose_movement_{m}_mean_{window_label}"
        if te in features and tb in features and cm in features and ns in features:
            aggs[f"Tail_end_relative_to_tail_base_centroid_nose_{m}_{window_label}"] = (
                features[te] - features[tb] - features[cm] - features[ns]
            )
    return aggs


def _compute_deviations(
    features: dict[str, Array1D],
    roll_windows: list[float],
    fps: float,
) -> dict[str, Array1D]:
    """Deviation features: mean(feature) - current(feature) value.

    SimBA uses explicit hard-coded deviation names that differ from the source
    key name (e.g. "Sum_euclid_distances_hull_deviation" not "_M1_M2_deviation").
    Also computes per-window rolling mean deviations.
    """
    aggs: dict[str, Array1D] = {}

    # Hard-coded SimBA deviation names
    deviation_map: dict[str, str] = {
        "Total_movement_all_bodyparts_both_mice": "Total_movement_all_bodyparts_both_mice_deviation",
        "Sum_euclidean_distance_hull_M1_M2": "Sum_euclid_distances_hull_deviation",
        "M1_hull_small_euclidean": "M1_smallest_euclid_distances_hull_deviation",
        "M1_hull_large_euclidean": "M1_largest_euclid_distances_hull_deviation",
        "M1_hull_mean_euclidean": "M1_mean_euclid_distances_hull_deviation",
        "M2_hull_small_euclidean": "M2_smallest_euclid_distances_hull_deviation",
        "M2_hull_large_euclidean": "M2_largest_euclid_distances_hull_deviation",
        "M2_hull_mean_euclidean": "M2_mean_euclid_distances_hull_deviation",
        "Centroid_distance": "Centroid_distance_deviation",
        "Total_angle_both_mice": "Total_angle_both_mice_deviation",
        "Movement_mouse_1_centroid": "Movement_mouse_1_deviation_centroid",
        "Movement_mouse_2_centroid": "Movement_mouse_2_deviation_centroid",
        "Mouse_1_poly_area": "Mouse_1_polygon_deviation",
        "Mouse_2_poly_area": "Mouse_2_polygon_deviation",
    }
    for src, dst in deviation_map.items():
        if src in features:
            aggs[dst] = features[src].mean() - features[src]

    if "Sum_probabilities" in features:
        aggs["Sum_probabilities_deviation"] = (
            features["Sum_probabilities"].mean() - features["Sum_probabilities"]
        )

    # Rolling mean deviations (SimBA loops over windows for each source)
    rolling_dev_prefixes = {
        "Total_movement_all_bodyparts_both_mice": "Total_movement_all_bodyparts_both_mice",
        "Sum_euclidean_distance_hull_M1_M2": "Sum_euclid_distances_hull",
        "M1_hull_small_euclidean": "Mouse1_smallest_euclid_distances",
        "M1_hull_large_euclidean": "Mouse1_largest_euclid_distances",
        "M1_hull_mean_euclidean": "Mouse1_mean_euclid_distances",
        "M2_hull_small_euclidean": "Mouse2_smallest_euclid_distances",
        "M2_hull_large_euclidean": "Mouse2_largest_euclid_distances",
        "M2_hull_mean_euclidean": "Mouse2_mean_euclid_distances",
    }

    # Additional rolling sources for Movement_mean_{w}_deviation
    rolling_movement_prefixes = {
        "Total_movement_centroids": "Movement",
    }

    for w in roll_windows:
        wl = str(w)
        for prefix in rolling_dev_prefixes.values():
            col = f"{prefix}_mean_{wl}"
            if col in features:
                aggs[f"{col}_deviation"] = features[col].mean() - features[col]
        for prefix in rolling_movement_prefixes.values():
            col = f"{prefix}_mean_{wl}"
            if col in features:
                aggs[f"{col}_deviation"] = features[col].mean() - features[col]

    return aggs


def _compute_tortuosities(
    arr_m1: Array2D,
    arr_m2: Array2D,
    roll_windows: list[float],
    fps: float,
) -> dict[str, Array1D]:
    """Path tortuosity for each animal's centroid movement."""
    aggs: dict[str, Array1D] = {}
    c1x, c1y = _get_xy(arr_m1, "Center")
    c2x, c2y = _get_xy(arr_m2, "Center")
    for w in roll_windows:
        wf = max(2, int(fps / w))
        wl = str(w)  # SimBA: raw float string
        aggs[f"Tortuosity_Mouse1_{wl}"] = _tortuosity(c1x, c1y, wf)
        aggs[f"Tortuosity_Mouse2_{wl}"] = _tortuosity(c2x, c2y, wf)
    return aggs


def _compute_percentile_ranks(
    features: dict[str, Array1D],
) -> dict[str, Array1D]:
    """Percentile rank features (replicating pandas .rank(pct=True))."""
    aggs: dict[str, Array1D] = {}

    def _pct_rank(vals: Array1D) -> Array1D:
        n = len(vals)
        if n <= 1:
            return np.ones(n, dtype=np.float64)
        order = np.argsort(vals)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64) / n
        return ranks

    # Rank sources use SimBA's exact movement names
    rank_sources = [
        "Movement_mouse_1_centroid",
        "Movement_mouse_2_centroid",
        "Centroid_distance",
        "Total_movement_all_bodyparts_both_mice",
    ]
    for key in rank_sources:
        if key in features:
            aggs[f"{key}_percentile_rank"] = _pct_rank(features[key])

    # Deviation percentile ranks (use SimBA's deviation names)
    dev_src_map = {
        "Movement_mouse_1_centroid": "Movement_mouse_1_deviation_centroid",
        "Movement_mouse_2_centroid": "Movement_mouse_2_deviation_centroid",
        "Centroid_distance": "Centroid_distance_deviation",
        "Total_movement_all_bodyparts_both_mice": "Total_movement_all_bodyparts_both_mice_deviation",
    }
    for dev_key in dev_src_map.values():
        if dev_key in features:
            aggs[f"{dev_key}_percentile_rank"] = _pct_rank(features[dev_key])

    if "Sum_probabilities_deviation" in features:
        aggs["Sum_probabilities_deviation_percentile_rank"] = _pct_rank(
            features["Sum_probabilities_deviation"],
        )

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
