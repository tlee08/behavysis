"""Feature extraction from preprocessed keypoints using SimBA feature math.

Replicates the SimBA ExtractFeaturesFrom16bps pipeline natively in
Polars + NumPy + SciPy.
No separate conda environment or subprocess required.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger
from scipy.spatial import ConvexHull

from behavysis.constants.bodypoints import BPMAP_SIMBA, INDIVS_SIMBA
from behavysis.models import ExperimentConfig, ExperimentMetadata
from behavysis.schemas import check_bpts_exist


def extract_features(
    keypoints_df: pl.DataFrame,
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
) -> pl.DataFrame:
    """Extract SimBA-compatible features from preprocessed keypoints.

    Parameters
    ----------
    keypoints_fp : Path
        Preprocessed keypoints filepath.
    features_fp : Path
        Filepath to save extracted_features dataframe.
    config_fp : Path
        Config JSON filepath.
    overwrite : bool
        Whether to overwrite the features_fp file if it exists.
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


#################################################
# Calculating SimBA features
# https://github.com/sgoldenlab/simba
#################################################


# ═══════════════════════════════════════════════════════════════════════════════
# Body-part naming conventions
# ═══════════════════════════════════════════════════════════════════════════════

SIMBA_BODY_PARTS = [
    "Ear_left",
    "Ear_right",
    "Nose",
    "Center",
    "Lat_left",
    "Lat_right",
    "Tail_base",
    "Tail_end",
]

SIMBA_FEATURE_NAMES = [
    "Mouse_1_nose_to_tail",
    "Mouse_2_nose_to_tail",
    "Mouse_1_width",
    "Mouse_2_width",
    "Mouse_1_Ear_distance",
    "Mouse_2_Ear_distance",
    "Mouse_1_Nose_to_centroid",
    "Mouse_2_Nose_to_centroid",
    "Mouse_1_Nose_to_lateral_left",
    "Mouse_2_Nose_to_lateral_left",
    "Mouse_1_Nose_to_lateral_right",
    "Mouse_2_Nose_to_lateral_right",
    "Mouse_1_Centroid_to_lateral_left",
    "Mouse_2_Centroid_to_lateral_left",
    "Mouse_1_Centroid_to_lateral_right",
    "Mouse_2_Centroid_to_lateral_right",
    "Centroid_distance",
    "Nose_to_nose_distance",
    "M1_Nose_to_M2_lat_left",
    "M1_Nose_to_M2_lat_right",
    "M2_Nose_to_M1_lat_left",
    "M2_Nose_to_M1_lat_right",
    "M1_Nose_to_M2_tail_base",
    "M2_Nose_to_M1_tail_base",
]

ROLL_WINDOW_DIVISORS = [2.0, 5.0, 7.5, 15, 30, 60, 90, 120]

# ═══════════════════════════════════════════════════════════════════════════════
# Index mapping for wide arrays
# ═══════════════════════════════════════════════════════════════════════════════

# Each body-part has x and y → 2 values per body-part, 8 body-parts = 16 columns
# Column layout for one animal's (n_frames, 16) array:
#   [Ear_left_x, Ear_left_y, Ear_right_x, Ear_right_y,
#    Nose_x, Nose_y, Center_x, Center_y,
#    Lat_left_x, Lat_left_y, Lat_right_x, Lat_right_y,
#    Tail_base_x, Tail_base_y, Tail_end_x, Tail_end_y]

# Taken from here:
# - https://github.com/sgoldenlab/simba/blob/master/simba/pose_configurations/configuration_names/pose_config_names.csv
# - https://github.com/sgoldenlab/simba/blob/master/simba/pose_configurations/bp_names/bp_names.csv
# 2 animals; 16 body-parts.

BP_XY_IDX = {
    "Ear_left": (0, 1),
    "Ear_right": (2, 3),
    "Nose": (4, 5),
    "Center": (6, 7),
    "Lat_left": (8, 9),
    "Lat_right": (10, 11),
    "Tail_base": (12, 13),
    "Tail_end": (14, 15),
}

# ═══════════════════════════════════════════════════════════════════════════════
# Typing aliases
# ═══════════════════════════════════════════════════════════════════════════════

type Array1D = np.ndarray[tuple[int], np.dtype[np.float64]]
type Array2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type Array3D = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]

# ═══════════════════════════════════════════════════════════════════════════════
# Vectorized math helpers (replicating SimBA's numba-jitted functions)
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
        degrees(atan2(|tail_x-cx|, |tail_y-cy|)
        - atan2(|nose_x-cx|, |nose_y-cy|))
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
    if points.shape[0] < 3:
        return 0.0, 0.0
    try:
        hull = ConvexHull(points)
        perimeter_px = hull.area  # In 2D, ConvexHull.area = perimeter
        area_px2 = hull.volume  # In 2D, ConvexHull.volume = area
        return perimeter_px / px_per_mm, area_px2 / (px_per_mm**2)
    except Exception:
        return 0.0, 0.0


def _cdist(points: Array2D) -> Array2D:
    """Pairwise Euclidean distances between all points (like scipy cdist)."""
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def _count_in_ranges(values: Array2D, ranges: list[tuple[float, float]]) -> Array2D:
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
    if n < 3 or window_frames < 2:
        return np.zeros(n, dtype=np.float64)

    window_samples = int(window_frames)
    window_samples = max(window_samples, 3)
    window_samples = min(window_samples, n)

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


def _rolling_stats(
    series: Array1D,
    windows: list[float],
    fps: float,
    *,
    center: bool = True,
) -> dict[str, Array1D]:
    """Compute rolling median, mean, sum for each window size.

    Returns dict of {f"{name}_{window_label}": array, ...}.
    """
    results = {}
    for w in windows:
        window_frames = max(2, int(fps / w))
        label = str(w).replace(".0", "")
        if window_frames > len(series):
            continue

        roll = (
            pl.Series(series)
            .rolling_median(
                window_size=window_frames,
                min_samples=1,
                center=center,
            )
            .to_numpy()
        )
        results[f"_median_{label}"] = roll

        roll = (
            pl.Series(series)
            .rolling_mean(
                window_size=window_frames,
                min_samples=1,
                center=center,
            )
            .to_numpy()
        )
        results[f"_mean_{label}"] = roll

        roll = (
            pl.Series(series)
            .rolling_sum(
                window_size=window_frames,
                min_samples=1,
                center=center,
            )
            .to_numpy()
        )
        results[f"_sum_{label}"] = roll

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main feature extraction
# ═══════════════════════════════════════════════════════════════════════════════


def compute_simba_features(
    keypoints_df: pl.DataFrame,
    fps: float,
    px_per_mm: float,
) -> pl.DataFrame:
    """Compute SimBA features from Polars long-form keypoints.

    Parameters
    ----------
    keypoints_df : pl.DataFrame
        Long-form KEYPOINTS_SCHEMA DataFrame.
    fps : float
        Frames per second.
    px_per_mm : float
        Pixels per mm scale factor.

    Returns:
    -------
    pl.DataFrame
        Wide features DataFrame with frame index and ~400+ feature columns.
        Column names match SimBA output exactly for downstream ML compatibility.
    """
    n_frames = keypoints_df.select("frame").n_unique()

    # Determine valid rolling windows (must be ≤ n_frames / 2)
    roll_windows = []
    for d in ROLL_WINDOW_DIVISORS:
        w = max(2, int(fps / d))
        if w <= n_frames / 2:
            roll_windows.append(d)

    # ── Build wide arrays per animal ──
    arr_m1, arr_m2, arr_prob = _pivot_to_wide(keypoints_df)

    # ── Compute all features ──
    features = {}

    features.update(_compute_distances(arr_m1, arr_m2, px_per_mm))
    features.update(_compute_movements(arr_m1, arr_m2, px_per_mm))
    features.update(_compute_hull(arr_m1, arr_m2, px_per_mm))
    features.update(_compute_cdist_features(arr_m1, arr_m2, px_per_mm))
    features.update(_compute_aggregates(features))
    features.update(_compute_angles(arr_m1, arr_m2, features))
    features.update(_compute_probability(arr_prob, features))
    features.update(_compute_rolling(features, roll_windows, fps))
    features.update(_compute_deviations(features))
    features.update(_compute_percentile_ranks(features))
    features.update(_compute_tortuosities(arr_m1, arr_m2, roll_windows, fps, features))

    return _build_output_df(keypoints_df, features)


# ═══════════════════════════════════════════════════════════════════════════════
# Pivot: long-form Polars → wide numpy arrays
# ═══════════════════════════════════════════════════════════════════════════════


def _pivot_to_wide(
    keypoints_df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert long-form keypoints to wide numpy arrays per animal.

    Returns (arr_m1, arr_m2, arr_prob):
        arr_m1: (n_frames, 16) — x,y for 8 body-parts of animal 1
        arr_m2: (n_frames, 16) — x,y for 8 body-parts of animal 2
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

    # Forward-fill then backward-fill any NaN positions
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
    # Backward fill
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
    """Compute all inter-body-part Euclidean distance features."""
    features = {}
    # Animal 1
    features["Mouse_1_nose_to_tail"] = _add_dist(
        arr_m1,
        arr_m1,
        "Nose",
        "Tail_base",
        px_per_mm,
    )
    features["Mouse_1_width"] = _add_dist(
        arr_m1,
        arr_m1,
        "Lat_left",
        "Lat_right",
        px_per_mm,
    )
    features["Mouse_1_Ear_distance"] = _add_dist(
        arr_m1,
        arr_m1,
        "Ear_left",
        "Ear_right",
        px_per_mm,
    )
    features["Mouse_1_Nose_to_centroid"] = _add_dist(
        arr_m1,
        arr_m1,
        "Nose",
        "Center",
        px_per_mm,
    )
    features["Mouse_1_Nose_to_lateral_left"] = _add_dist(
        arr_m1,
        arr_m1,
        "Nose",
        "Lat_left",
        px_per_mm,
    )
    features["Mouse_1_Nose_to_lateral_right"] = _add_dist(
        arr_m1,
        arr_m1,
        "Nose",
        "Lat_right",
        px_per_mm,
    )
    features["Mouse_1_Centroid_to_lateral_left"] = _add_dist(
        arr_m1,
        arr_m1,
        "Center",
        "Lat_left",
        px_per_mm,
    )
    features["Mouse_1_Centroid_to_lateral_right"] = _add_dist(
        arr_m1,
        arr_m1,
        "Center",
        "Lat_right",
        px_per_mm,
    )

    # Animal 2
    features["Mouse_2_nose_to_tail"] = _add_dist(
        arr_m2,
        arr_m2,
        "Nose",
        "Tail_base",
        px_per_mm,
    )
    features["Mouse_2_width"] = _add_dist(
        arr_m2,
        arr_m2,
        "Lat_left",
        "Lat_right",
        px_per_mm,
    )
    features["Mouse_2_Ear_distance"] = _add_dist(
        arr_m2,
        arr_m2,
        "Ear_left",
        "Ear_right",
        px_per_mm,
    )
    features["Mouse_2_Nose_to_centroid"] = _add_dist(
        arr_m2,
        arr_m2,
        "Nose",
        "Center",
        px_per_mm,
    )
    features["Mouse_2_Nose_to_lateral_left"] = _add_dist(
        arr_m2,
        arr_m2,
        "Nose",
        "Lat_left",
        px_per_mm,
    )
    features["Mouse_2_Nose_to_lateral_right"] = _add_dist(
        arr_m2,
        arr_m2,
        "Nose",
        "Lat_right",
        px_per_mm,
    )
    features["Mouse_2_Centroid_to_lateral_left"] = _add_dist(
        arr_m2,
        arr_m2,
        "Center",
        "Lat_left",
        px_per_mm,
    )
    features["Mouse_2_Centroid_to_lateral_right"] = _add_dist(
        arr_m2,
        arr_m2,
        "Center",
        "Lat_right",
        px_per_mm,
    )

    # Cross-animal
    features["Centroid_distance"] = _add_dist(
        arr_m1,
        arr_m2,
        "Center",
        "Center",
        px_per_mm,
    )
    features["Nose_to_nose_distance"] = _add_dist(
        arr_m1,
        arr_m2,
        "Nose",
        "Nose",
        px_per_mm,
    )
    features["M1_Nose_to_M2_lat_left"] = _add_dist(
        arr_m1,
        arr_m2,
        "Nose",
        "Lat_left",
        px_per_mm,
    )
    features["M1_Nose_to_M2_lat_right"] = _add_dist(
        arr_m1,
        arr_m2,
        "Nose",
        "Lat_right",
        px_per_mm,
    )
    features["M2_Nose_to_M1_lat_left"] = _add_dist(
        arr_m2,
        arr_m1,
        "Nose",
        "Lat_left",
        px_per_mm,
    )
    features["M2_Nose_to_M1_lat_right"] = _add_dist(
        arr_m2,
        arr_m1,
        "Nose",
        "Lat_right",
        px_per_mm,
    )
    features["M1_Nose_to_M2_tail_base"] = _add_dist(
        arr_m1,
        arr_m2,
        "Nose",
        "Tail_base",
        px_per_mm,
    )
    features["M2_Nose_to_M1_tail_base"] = _add_dist(
        arr_m2,
        arr_m1,
        "Nose",
        "Tail_base",
        px_per_mm,
    )

    # Return
    return features


def _add_dist(
    arr_a: Array2D,
    arr_b: Array2D,
    bp_a: str,
    bp_b: str,
    px_per_mm: float,
) -> Array1D:
    ax, ay = _get_xy(arr_a, bp_a)
    bx, by = _get_xy(arr_b, bp_b)
    return _euclidean(ax, ay, bx, by, px_per_mm)


def _compute_movements(
    arr_m1: Array2D,
    arr_m2: Array2D,
    px_per_mm: float,
) -> dict[str, Array1D]:
    """Frame-to-frame movement for each body-part."""
    features = {}
    for simba_bp in SIMBA_BODY_PARTS:
        ix, iy = BP_XY_IDX[simba_bp]
        features[f"Movement_mouse_1_{simba_bp.lower()}"] = _movement_bp(
            arr_m1[:, ix],
            arr_m1[:, iy],
            px_per_mm,
        )
        features[f"Movement_mouse_2_{simba_bp.lower()}"] = _movement_bp(
            arr_m2[:, ix],
            arr_m2[:, iy],
            px_per_mm,
        )
    # Return
    return features


def _compute_hull(
    arr_m1: Array2D,
    arr_m2: Array2D,
    px_per_mm: float,
) -> dict[str, Array1D]:
    """Convex hull perimeter and area features."""
    return {
        **_add_hull(arr_m1, px_per_mm, "M1"),
        **_add_hull(arr_m2, px_per_mm, "M2"),
    }


def _add_hull(arr: Array2D, px_per_mm: float, label: str) -> dict[str, Array1D]:
    n = arr.shape[0]
    perimeters = np.zeros(n, dtype=np.float64)
    areas = np.zeros(n, dtype=np.float64)

    for i in range(n):
        points = arr[i].reshape(-1, 2)
        valid = ~np.isnan(points).any(axis=1)
        if valid.sum() >= 3:
            perimeters[i], areas[i] = _hull_perimeter(points[valid], px_per_mm)

    features = {}
    features[f"Mouse_{label.lower()}_poly_area"] = perimeters
    # Polygon size change: frame-to-frame delta of area
    change = np.zeros(n, dtype=np.float64)
    change[1:] = np.abs(areas[1:] - areas[:-1])
    features[f"Mouse_{label.lower()}_polygon_size_change"] = change
    # Return
    return features


def _compute_cdist_features(
    arr_m1: Array2D,
    arr_m2: Array2D,
    px_per_mm: float,
) -> dict[str, Array1D]:
    """Pairwise distance statistics within each animal's hull."""
    _add_cdist(arr_m1, px_per_mm, "M1")
    _add_cdist(arr_m2, px_per_mm, "M2")

    # Sum of both hull sums
    if "M1_hull_sum_euclidean" in features and "M2_hull_sum_euclidean" in features:
        features["Sum_euclidean_distance_hull_M1_M2"] = (
            features["M1_hull_sum_euclidean"] + features["M2_hull_sum_euclidean"]
        )


def _add_cdist(arr: Array2D, px_per_mm: float, label: str) -> dict[str, Array1D]:
    n = arr.shape[0]
    large = np.zeros(n, dtype=np.float64)
    small = np.zeros(n, dtype=np.float64)
    mean_ = np.zeros(n, dtype=np.float64)
    sum_ = np.zeros(n, dtype=np.float64)

    for i in range(n):
        points = arr[i].reshape(-1, 2)
        valid = ~np.isnan(points).any(axis=1)
        pts = points[valid]
        if len(pts) >= 2:
            dists = _cdist(pts)
            triu = dists[np.triu_indices_from(dists, k=1)]
            if len(triu) > 0:
                large[i] = np.max(triu) / px_per_mm
                small[i] = np.min(triu) / px_per_mm
                mean_[i] = np.mean(triu) / px_per_mm
                sum_[i] = np.sum(triu) / px_per_mm

    return {
        f"{label}_hull_large_euclidean": large,
        f"{label}_hull_small_euclidean": small,
        f"{label}_hull_mean_euclidean": mean_,
        f"{label}_hull_sum_euclidean": sum_,
    }


def _compute_aggregates(features: dict[str, Array1D]):
    """Sum aggregate movement features."""
    # Total movement centroids
    if "Movement_mouse_1_center" in features and "Movement_mouse_2_center" in features:
        features["Total_movement_centroids"] = (
            features["Movement_mouse_1_center"] + features["Movement_mouse_2_center"]
        )

    # Total movement tail ends
    if (
        "Movement_mouse_1_tail_end" in features
        and "Movement_mouse_2_tail_end" in features
    ):
        features["Total_movement_tail_ends"] = (
            features["Movement_mouse_1_tail_end"]
            + features["Movement_mouse_2_tail_end"]
        )

    # Total movement all bodyparts per animal
    bp_names = [b.lower() for b in SIMBA_BODY_PARTS]
    m1_keys = [f"Movement_mouse_1_{bp}" for bp in bp_names]
    m2_keys = [f"Movement_mouse_2_{bp}" for bp in bp_names]

    m1_total = np.zeros_like(features.get(m1_keys[0], np.zeros(1)))
    m2_total = np.zeros_like(m1_total)
    for k in m1_keys:
        if k in features:
            m1_total += features[k]
    for k in m2_keys:
        if k in features:
            m2_total += features[k]

    features["Total_movement_all_bodyparts_M1"] = m1_total
    features["Total_movement_all_bodyparts_M2"] = m2_total
    features["Total_movement_all_bodyparts_both_mice"] = m1_total + m2_total


def _compute_angles(arr_m1, arr_m2, features):
    """3-point angle (nose→center→tail_base) per animal."""
    n1x, n1y = _get_xy(arr_m1, "Nose")
    c1x, c1y = _get_xy(arr_m1, "Center")
    t1x, t1y = _get_xy(arr_m1, "Tail_base")
    features["Mouse_1_angle"] = _angle3pt_vectorized(n1x, n1y, c1x, c1y, t1x, t1y)

    n2x, n2y = _get_xy(arr_m2, "Nose")
    c2x, c2y = _get_xy(arr_m2, "Center")
    t2x, t2y = _get_xy(arr_m2, "Tail_base")
    features["Mouse_2_angle"] = _angle3pt_vectorized(n2x, n2y, c2x, c2y, t2x, t2y)

    features["Total_angle_both_mice"] = (
        features["Mouse_1_angle"] + features["Mouse_2_angle"]
    )


def _compute_probability(arr_prob, features):
    """Probability-based features: sum of likelihoods, low-prob detection counts."""
    features["Sum_probabilities"] = np.sum(arr_prob, axis=1)

    ranges = [(0.0, 0.1), (0.0, 0.5), (0.0, 0.75)]
    counts = _count_in_ranges(arr_prob, ranges)
    features["Low_prob_detections_0.1"] = counts[:, 0]
    features["Low_prob_detections_0.5"] = counts[:, 1]
    features["Low_prob_detections_0.75"] = counts[:, 2]


def _compute_rolling(features, roll_windows, fps):
    """Compute rolling window median/mean/sum for key features."""
    rolling_targets = [
        ("Sum_euclidean_distance_hull_M1_M2", "hull"),
        ("Total_movement_centroids", "Movement"),
        ("Centroid_distance", "Distance"),
        ("Mouse_1_width", "Mouse1_width"),
        ("Mouse_2_width", "Mouse2_width"),
        ("M1_hull_mean_euclidean", "mean_euclid_distances_M1"),
        ("M2_hull_mean_euclidean", "mean_euclid_distances_M2"),
        ("M1_hull_small_euclidean", "smallest_euclid_distances_M1"),
        ("M2_hull_small_euclidean", "smallest_euclid_distances_M2"),
        ("M1_hull_large_euclidean", "largest_euclid_distances_M1"),
        ("M2_hull_large_euclidean", "largest_euclid_distances_M2"),
        ("Total_movement_all_bodyparts_both_mice", "total_movement"),
        ("Movement_mouse_1_tail_base", "Tail_base_movement_M1"),
        ("Movement_mouse_2_tail_base", "Tail_base_movement_M2"),
        ("Movement_mouse_1_center", "Centroid_movement_M1"),
        ("Movement_mouse_2_center", "Centroid_movement_M2"),
        ("Movement_mouse_1_tail_end", "Tail_end_movement_M1"),
        ("Movement_mouse_2_tail_end", "Tail_end_movement_M2"),
        ("Movement_mouse_1_nose", "Nose_movement_M1"),
        ("Movement_mouse_2_nose", "Nose_movement_M2"),
    ]

    for src_key, prefix in rolling_targets:
        if src_key not in features:
            continue
        stats = _rolling_stats(features[src_key], roll_windows, fps)
        for suffix, arr in stats.items():
            features[f"{prefix}{suffix}"] = arr

    # Total angle rolling
    if "Total_angle_both_mice" in features:
        stats = _rolling_stats(features["Total_angle_both_mice"], roll_windows, fps)
        for suffix, arr in stats.items():
            features[f"Total_angle_both_mice{suffix}"] = arr

    # Relative tail end features
    for mouse in ["M1", "M2"]:
        _compute_tail_end_relative(mouse, features, roll_windows, fps)


def _compute_tail_end_relative(mouse, features, roll_windows, fps):
    """Tail_end movement relative to (tail_base + centroid + nose) movement."""
    ml = mouse.lower()
    tail_end_key = f"Movement_mouse_{ml}_tail_end"
    tail_base_key = f"Movement_mouse_{ml}_tail_base"
    centroid_key = f"Movement_mouse_{ml}_center"
    nose_key = f"Movement_mouse_{ml}_nose"

    if not all(
        k in features for k in [tail_end_key, tail_base_key, centroid_key, nose_key]
    ):
        return

    rel = features[tail_end_key] - (
        features[tail_base_key] + features[centroid_key] + features[nose_key]
    )
    features[f"Tail_end_relative_to_tail_base_centroid_nose_{mouse}"] = rel

    stats = _rolling_stats(rel, roll_windows, fps)
    for suffix, arr in stats.items():
        features[f"Tail_end_relative_to_tail_base_centroid_nose_{mouse}{suffix}"] = arr


def _compute_deviations(features):
    """Deviation features: mean(feature) - current(feature) value."""
    deviation_sources = [
        "Total_movement_all_bodyparts_both_mice",
        "Sum_euclidean_distance_hull_M1_M2",
        "M1_hull_small_euclidean",
        "M2_hull_small_euclidean",
        "M1_hull_large_euclidean",
        "M2_hull_large_euclidean",
        "M1_hull_mean_euclidean",
        "M2_hull_mean_euclidean",
        "Centroid_distance",
        "Total_angle_both_mice",
        "Movement_mouse_1_center",
        "Movement_mouse_2_center",
        "Mouse_1_poly_area",
        "Mouse_2_poly_area",
    ]

    for key in deviation_sources:
        if key not in features:
            continue
        vals = features[key]
        features[f"{key}_deviation"] = np.mean(vals) - vals

    # Sum_probabilities deviation
    if "Sum_probabilities" in features:
        vals = features["Sum_probabilities"]
        features["Sum_probabilities_deviation"] = np.mean(vals) - vals


def _compute_percentile_ranks(features):
    """Percentile rank features (replicating pandas .rank(pct=True))."""
    rank_sources = [
        "Movement_mouse_1_center",
        "Movement_mouse_2_center",
        "Centroid_distance",
        "Total_movement_all_bodyparts_both_mice",
    ]

    for key in rank_sources:
        if key not in features:
            continue
        vals = features[key]
        # Percentile rank: (rank - 1) / (n - 1) for 0-1 range
        n = len(vals)
        if n > 1:
            order = np.argsort(vals)
            ranks = np.empty(n, dtype=np.float64)
            ranks[order] = np.arange(1, n + 1, dtype=np.float64) / n
        else:
            ranks = np.ones(n, dtype=np.float64)
        features[f"{key}_percentile_rank"] = ranks

    # Deviation percentile ranks
    for key in rank_sources:
        dev_key = f"{key}_deviation"
        if dev_key not in features:
            continue
        vals = features[dev_key]
        n = len(vals)
        if n > 1:
            order = np.argsort(vals)
            ranks = np.empty(n, dtype=np.float64)
            ranks[order] = np.arange(1, n + 1, dtype=np.float64) / n
        else:
            ranks = np.ones(n, dtype=np.float64)
        features[f"{key}_deviation_percentile_rank"] = ranks

    # Sum_probabilities deviation percentile rank
    if "Sum_probabilities_deviation" in features:
        vals = features["Sum_probabilities_deviation"]
        n = len(vals)
        if n > 1:
            order = np.argsort(vals)
            ranks = np.empty(n, dtype=np.float64)
            ranks[order] = np.arange(1, n + 1, dtype=np.float64) / n
        else:
            ranks = np.ones(n, dtype=np.float64)
        features["Sum_probabilities_deviation_percentile_rank"] = ranks


def _compute_tortuosities(arr_m1, arr_m2, roll_windows, fps, features):
    """Path tortuosity for each animal's centroid movement."""
    c1x, c1y = _get_xy(arr_m1, "Center")
    c2x, c2y = _get_xy(arr_m2, "Center")

    for w in roll_windows:
        window_frames = max(2, int(fps / w))
        label = str(w).replace(".0", "")
        features[f"Tortuosity_Mouse1_{label}"] = _tortuosity(c1x, c1y, window_frames)
        features[f"Tortuosity_Mouse2_{label}"] = _tortuosity(c2x, c2y, window_frames)


def _build_output_df(keypoints_df, features):
    """Build final Polars DataFrame with frame index + all feature columns."""
    frames = keypoints_df.select("frame").unique().sort("frame").to_series().to_list()

    col_data = {"frame": frames}
    for name, arr in features.items():
        col_data[name] = arr

    return pl.DataFrame(col_data)
