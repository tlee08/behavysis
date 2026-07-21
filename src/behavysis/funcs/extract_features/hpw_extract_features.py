"""Feature extraction for hind paw withdrawal and rearing behaviours.

Single rat, side-on camera view, glass cylinder chamber.
Designed for DeepLabCut multi-animal keypoint tracking with 17 bodyparts
(plus 2 static arena markers).

Features are computed from KEYPOINTS_SCHEMA long-form DataFrames and
grouped into two behaviour-specific batteries:

- **Rearing** (R01-R06): vertical posture, body elongation, upward movement.
- **Hind Paw Withdrawal** (W01-W19): paw vertical/horizontal kinematics,
  elevation above estimated floor, heel-toe distance, paw asymmetry,
  and body-stillness control features that help distinguish isolated
  paw lifts from stepping/walking.

Coordinate convention (image space):
  x -> rightward positive, y -> downward positive.
  "Upward" movement in the world = y *decreases* in image space.
  Features report upward velocity/acceleration as *positive* values
  (i.e. we negate the raw image-space derivatives).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy.ndimage import (
    maximum_filter1d,
    minimum_filter1d,
    uniform_filter1d,
)

if TYPE_CHECKING:
    from behavysis.constants import Array1D, Array2D

# ═══════════════════════════════════════════════════════════════════════════════
# Bodypart name constants
# ═══════════════════════════════════════════════════════════════════════════════

ARENA_R = "arena_r"
ARENA_L = "arena_l"
NOSE = "nose"
EAR_R = "ear_r"
EAR_L = "ear_l"
FRONT_TOE_R = "front_toe_r"
FRONT_KNEE_R = "front_knee_r"
FRONT_TOE_L = "front_toe_l"
FRONT_KNEE_L = "front_knee_l"
HIND_TOE_R = "hind_toe_r"
HIND_HEEL_R = "hind_heel_r"
HIND_KNEE_R = "hind_knee_r"
HIND_TOE_L = "hind_toe_l"
HIND_HEEL_L = "hind_heel_l"
HIND_KNEE_L = "hind_knee_l"
MID_BACK = "mid_back"
LOWER_BACK = "lower_back"
TAIL_BASE = "tail_base"
TAIL_TIP = "tail_tip"

INDIVIDUAL = "rat"

ALL_BODYPARTS: list[str] = [
    NOSE,
    EAR_R,
    EAR_L,
    FRONT_TOE_R,
    FRONT_KNEE_R,
    FRONT_TOE_L,
    FRONT_KNEE_L,
    HIND_TOE_R,
    HIND_HEEL_R,
    HIND_KNEE_R,
    HIND_TOE_L,
    HIND_HEEL_L,
    HIND_KNEE_L,
    MID_BACK,
    LOWER_BACK,
    TAIL_BASE,
    TAIL_TIP,
]

# Static arena reference markers (not used in feature computation)
ARENA_BPTS = [ARENA_R, ARENA_L]

# Semantic bodypart groups for feature computation
HEAD_BPTS = [NOSE, EAR_R, EAR_L]
FRONT_PAW_R_BPTS = [FRONT_TOE_R, FRONT_KNEE_R]
FRONT_PAW_L_BPTS = [FRONT_TOE_L, FRONT_KNEE_L]
FRONT_PAW_ALL_BPTS = FRONT_PAW_R_BPTS + FRONT_PAW_L_BPTS
HIND_PAW_R_BPTS = [HIND_TOE_R, HIND_HEEL_R, HIND_KNEE_R]
HIND_PAW_L_BPTS = [HIND_TOE_L, HIND_HEEL_L, HIND_KNEE_L]
HIND_PAW_ALL_BPTS = HIND_PAW_R_BPTS + HIND_PAW_L_BPTS
BACK_BPTS = [MID_BACK, LOWER_BACK]
TAIL_BPTS = [TAIL_BASE, TAIL_TIP]

# ═══════════════════════════════════════════════════════════════════════════════
# Feature name constants — Rearing
# ═══════════════════════════════════════════════════════════════════════════════

R01_BACK_ANGLE_DEG = "R01_back_angle_deg"
R02_NOSE_ELEVATION_MM = "R02_nose_elevation_mm"
R03_BODY_ELONGATION_RATIO = "R03_body_elongation_ratio"
R04_CENTROID_VERTICAL_VELOCITY_MM_S = "R04_centroid_vertical_velocity_mm_s"
R05_FRONT_PAW_ELEVATION_MM = "R05_front_paw_elevation_mm"
R06_NOSE_VERTICAL_VELOCITY_MM_S = "R06_nose_vertical_velocity_mm_s"

REARING_FEATURES: list[str] = [
    R01_BACK_ANGLE_DEG,
    R02_NOSE_ELEVATION_MM,
    R03_BODY_ELONGATION_RATIO,
    R04_CENTROID_VERTICAL_VELOCITY_MM_S,
    R05_FRONT_PAW_ELEVATION_MM,
    R06_NOSE_VERTICAL_VELOCITY_MM_S,
]

# ═══════════════════════════════════════════════════════════════════════════════
# Feature name constants — Hind Paw Withdrawal
# ═══════════════════════════════════════════════════════════════════════════════

# --- Right paw ---
W01_R_PAW_VERTICAL_V_MM_S = "W01_r_paw_vertical_velocity_mm_s"
W02_R_PAW_HORIZONTAL_V_MM_S = "W02_r_paw_horizontal_velocity_mm_s"
W03_R_PAW_VY_TO_VX_RATIO = "W03_r_paw_vy_to_vx_ratio"
W04_R_PAW_ELEVATION_MM = "W04_r_paw_elevation_mm"
W05_R_PAW_HEEL_TOE_DIST_MM = "W05_r_paw_heel_toe_dist_mm"
W06_R_PAW_VERTICAL_A_MM_S2 = "W06_r_paw_vertical_accel_mm_s2"

# --- Left paw ---
W07_L_PAW_VERTICAL_V_MM_S = "W07_l_paw_vertical_velocity_mm_s"
W08_L_PAW_HORIZONTAL_V_MM_S = "W08_l_paw_horizontal_velocity_mm_s"
W09_L_PAW_VY_TO_VX_RATIO = "W09_l_paw_vy_to_vx_ratio"
W10_L_PAW_ELEVATION_MM = "W10_l_paw_elevation_mm"
W11_L_PAW_HEEL_TOE_DIST_MM = "W11_l_paw_heel_toe_dist_mm"
W12_L_PAW_VERTICAL_A_MM_S2 = "W12_l_paw_vertical_accel_mm_s2"

# --- Asymmetry & control ---
W13_PAW_ELEVATION_ASYMMETRY_MM = "W13_paw_elevation_asymmetry_mm"
W14_PAW_VERTICAL_V_ASYMMETRY_MM_S = "W14_paw_vertical_velocity_asymmetry_mm_s"
W15_HIND_BODY_VERTICAL_V_MM_S = "W15_hind_body_vertical_velocity_mm_s"
W16_R_PAW_RELATIVE_VERTICAL_V_MM_S = "W16_r_paw_relative_vertical_velocity_mm_s"
W17_L_PAW_RELATIVE_VERTICAL_V_MM_S = "W17_l_paw_relative_vertical_velocity_mm_s"
W18_R_PAW_VY_PEAK_MM_S = "W18_r_paw_vy_peak_mm_s"
W19_L_PAW_VY_PEAK_MM_S = "W19_l_paw_vy_peak_mm_s"

WITHDRAWAL_FEATURES: list[str] = [
    W01_R_PAW_VERTICAL_V_MM_S,
    W02_R_PAW_HORIZONTAL_V_MM_S,
    W03_R_PAW_VY_TO_VX_RATIO,
    W04_R_PAW_ELEVATION_MM,
    W05_R_PAW_HEEL_TOE_DIST_MM,
    W06_R_PAW_VERTICAL_A_MM_S2,
    W07_L_PAW_VERTICAL_V_MM_S,
    W08_L_PAW_HORIZONTAL_V_MM_S,
    W09_L_PAW_VY_TO_VX_RATIO,
    W10_L_PAW_ELEVATION_MM,
    W11_L_PAW_HEEL_TOE_DIST_MM,
    W12_L_PAW_VERTICAL_A_MM_S2,
    W13_PAW_ELEVATION_ASYMMETRY_MM,
    W14_PAW_VERTICAL_V_ASYMMETRY_MM_S,
    W15_HIND_BODY_VERTICAL_V_MM_S,
    W16_R_PAW_RELATIVE_VERTICAL_V_MM_S,
    W17_L_PAW_RELATIVE_VERTICAL_V_MM_S,
    W18_R_PAW_VY_PEAK_MM_S,
    W19_L_PAW_VY_PEAK_MM_S,
]

ALL_HPW_FEATURES: list[str] = REARING_FEATURES + WITHDRAWAL_FEATURES

# ═══════════════════════════════════════════════════════════════════════════════
# Smoothing / derivative parameters (hardcoded for MWE)
# ═══════════════════════════════════════════════════════════════════════════════

_POS_SMOOTH_WINDOW: int = 3  # frames — light smoothing of raw positions
_VEL_SMOOTH_WINDOW: int = 3  # frames — smoothing before computing acceleration
_FLOOR_ROLL_WINDOW_SEC: float = 5.0  # seconds for rolling floor estimate
_VY_PEAK_WINDOW_SEC: float = 0.2  # seconds for local vy peak detection
_EPS: float = 1e-6  # small constant for safe division
_MIN_KURT_WINDOW: int = 4  # minimum window frames for valid kurtosis

# ═══════════════════════════════════════════════════════════════════════════════
# Rolling window parameters
# ═══════════════════════════════════════════════════════════════════════════════

ROLL_WINDOW_DIVISORS: list[float] = [2, 5, 6, 7.5, 15]

# ═══════════════════════════════════════════════════════════════════════════════
# Cross-feature name constants
# ═══════════════════════════════════════════════════════════════════════════════

X01_BODY_STILLNESS_MM_S = "X01_body_stillness_mm_s"
X02_R_PAW_LIFT_RATIO = "X02_r_paw_lift_ratio"
X03_L_PAW_LIFT_RATIO = "X03_l_paw_lift_ratio"

CROSS_FEATURES: list[str] = [
    X01_BODY_STILLNESS_MM_S,
    X02_R_PAW_LIFT_RATIO,
    X03_L_PAW_LIFT_RATIO,
]

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers — DataFrame → numpy extraction
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_bodypart_xy(
    keypoints_df: pl.DataFrame,
    bodypart: str,
    individual: str = INDIVIDUAL,
) -> tuple[Array1D, Array1D, int]:
    """Extract (x, y) numpy arrays for a single bodypart, sorted by frame.

    Returns (x, y, n_frames). Gaps in frame coverage are filled with NaN
    and then forward/backward filled.
    """
    bp_df = (
        keypoints_df.filter(
            pl.col("individual") == individual,
            pl.col("bodypart") == bodypart,
        )
        .select(["frame", "x", "y"])
        .sort("frame")
    )

    frames_all = (
        keypoints_df.select("frame").unique().sort("frame").to_series().to_numpy()
    )
    n_frames = len(frames_all)

    frame_to_idx = {int(f): i for i, f in enumerate(frames_all)}
    bp_frames = bp_df.select("frame").to_series().to_numpy()
    bp_x = bp_df.select("x").to_series().to_numpy()
    bp_y = bp_df.select("y").to_series().to_numpy()

    x_arr = np.full(n_frames, np.nan, dtype=np.float64)
    y_arr = np.full(n_frames, np.nan, dtype=np.float64)
    for i, f in enumerate(bp_frames):
        idx = frame_to_idx[int(f)]
        x_arr[idx] = bp_x[i]
        y_arr[idx] = bp_y[i]

    x_arr = _ffill_bfill_1d(x_arr)
    y_arr = _ffill_bfill_1d(y_arr)
    return x_arr, y_arr, n_frames


def _get_bodypart_xy_dict(
    keypoints_df: pl.DataFrame,
    bodyparts: list[str],
    individual: str = INDIVIDUAL,
) -> dict[str, tuple[Array1D, Array1D]]:
    """Return {bodypart: (x_array, y_array)} for a list of bodyparts."""
    result: dict[str, tuple[Array1D, Array1D]] = {}
    for bp in bodyparts:
        x_arr, y_arr, _ = _extract_bodypart_xy(keypoints_df, bp, individual)
        result[bp] = (x_arr, y_arr)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers — NaN handling
# ═══════════════════════════════════════════════════════════════════════════════


def _ffill_bfill_1d(arr: Array1D) -> Array1D:
    """Forward-fill then backward-fill NaN values in a 1D array."""
    mask = np.isnan(arr)
    if not mask.any():
        return arr
    idx = np.where(~mask, np.arange(len(arr)), 0)
    np.maximum.accumulate(idx, out=idx)
    arr = arr[idx]
    mask = np.isnan(arr)
    if mask.any():
        idx = np.where(~mask, np.arange(len(arr)), len(arr) - 1)
        np.minimum.accumulate(idx[::-1], out=idx[::-1])
        arr = arr[idx]
    return arr


def _ffill_bfill_2d(arr: Array2D) -> Array2D:
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
# Helpers — kinematics
# ═══════════════════════════════════════════════════════════════════════════════


def _smooth_uniform(arr: Array1D, window: int) -> Array1D:
    """Apply uniform (boxcar) smoothing to a 1D array."""
    if window <= 1:
        return arr.copy()
    return uniform_filter1d(arr.astype(np.float64), size=window, mode="nearest")


def _vertical_velocity(
    y: Array1D,
    px_per_mm: float,
    fps: float,
    smooth_window: int = _POS_SMOOTH_WINDOW,
) -> Array1D:
    """Upward velocity in mm/s from y-coordinate array.

    Positive = upward movement (y decreases in image space → we negate diff).
    First frame is padded with 0 (no movement before start).
    """
    y_smooth = _smooth_uniform(y, smooth_window)
    vy_raw = -np.diff(y_smooth, prepend=y_smooth[0])  # negate: upward = positive
    return vy_raw / px_per_mm * fps


def _horizontal_velocity(
    x: Array1D,
    px_per_mm: float,
    fps: float,
    smooth_window: int = _POS_SMOOTH_WINDOW,
) -> Array1D:
    """Horizontal velocity in mm/s from x-coordinate array.

    Positive = rightward movement.
    """
    x_smooth = _smooth_uniform(x, smooth_window)
    vx_raw = np.diff(x_smooth, prepend=x_smooth[0])
    return vx_raw / px_per_mm * fps


def _acceleration_from_velocity(
    v: Array1D,
    fps: float,
    smooth_window: int = _VEL_SMOOTH_WINDOW,
) -> Array1D:
    """Acceleration from velocity array, in units/s²."""
    v_smooth = _smooth_uniform(v, smooth_window)
    a_raw = np.diff(v_smooth, prepend=v_smooth[0])
    return a_raw * fps


def _local_peak(
    signal: Array1D,
    window_frames: int,
) -> Array1D:
    """Per-frame local maximum of signal within ±window_frames.

    Produces a smoothed envelope that captures local peak intensity
    of the signal — useful for detecting transient spikes (e.g. paw flinch).
    """
    half = window_frames // 2
    n = len(signal)
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result[i] = np.max(signal[lo:hi])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers — geometry
# ═══════════════════════════════════════════════════════════════════════════════


def _angle_from_horizontal_deg(
    x: Array1D,
    y: Array1D,
) -> Array1D:
    """Angle of the line defined by (x, y) vectors from horizontal, in degrees.

    Treats (x, y) as a vector from origin. Returns [-180, 180]:
      0 deg = horizontal right, 90 deg = straight up.
    For a two-point line (x1,y1)->(x2,y2), pass dx=x2-x1, dy=y2-y1.
    """
    return np.degrees(np.arctan2(-y, x))


def _euclidean(
    ax: Array1D,
    ay: Array1D,
    bx: Array1D,
    by: Array1D,
    px_per_mm: float,
) -> Array1D:
    """Euclidean distance between two point sets, scaled to mm."""
    return np.hypot(ax - bx, ay - by) / px_per_mm


# ═══════════════════════════════════════════════════════════════════════════════
# Floor estimate
# ═══════════════════════════════════════════════════════════════════════════════


def _estimate_floor_y(
    hind_paw_y_arrays: list[Array1D],
    fps: float,
) -> Array1D:
    """Estimate floor y-position (image coords, y ↑ downward).

    Takes all hind paw bodypart y-arrays, computes a per-frame robust
    maximum (90th percentile across bodyparts), then applies a heavy
    rolling mean to produce a slowly-moving floor reference.
    """
    stacked = np.column_stack(hind_paw_y_arrays)
    floor_raw = np.percentile(stacked, 90, axis=1)
    roll_frames = max(1, int(fps * _FLOOR_ROLL_WINDOW_SEC))
    floor = _smooth_uniform(floor_raw, roll_frames)
    floor[:roll_frames] = floor[roll_frames]
    return floor


# ═══════════════════════════════════════════════════════════════════════════════
# Rolling window aggregation
# ═══════════════════════════════════════════════════════════════════════════════


def _rolling_window_stats(
    arr: Array1D,
    window: int,
) -> dict[str, Array1D]:
    """Compute rolling mean, std, min, max, and excess kurtosis.

    Uses scipy.ndimage filters — vectorized, centered windows,
    ``mode="nearest"`` for edge handling (no NaN at boundaries).

    Parameters
    ----------
    arr : Array1D
        Input 1D feature array.
    window : int
        Rolling window size in frames.

    Returns:
    -------
    dict[str, Array1D]
        ``{"_mean", "_std", "_min", "_max", "_kurt"}`` arrays.
    """
    m = uniform_filter1d(arr, size=window, mode="nearest")
    lo = minimum_filter1d(arr, size=window, mode="nearest")
    hi = maximum_filter1d(arr, size=window, mode="nearest")

    m2 = uniform_filter1d(np.square(arr), size=window, mode="nearest")
    var = np.maximum(m2 - np.square(m), 0.0)
    std = np.sqrt(var)

    # Excess kurtosis: E[(X-u)^4] / sigma^4 - 3
    # Requires >= _MIN_KURT_WINDOW samples; zeroed for smaller windows.
    if window >= _MIN_KURT_WINDOW:
        m3 = uniform_filter1d(np.power(arr, 3), size=window, mode="nearest")
        m4 = uniform_filter1d(np.power(arr, 4), size=window, mode="nearest")
        mu4 = m4 - 4 * m * m3 + 6 * np.square(m) * m2 - 3 * np.power(m, 4)
        kurt = np.divide(
            mu4,
            np.square(var) + _EPS,
            out=np.zeros_like(mu4, dtype=np.float64),
        ) - 3.0
    else:
        kurt = np.zeros_like(m)

    return {"_mean": m, "_std": std, "_min": lo, "_max": hi, "_kurt": kurt}


def _compute_rolling_aggregates(
    features: dict[str, Array1D],
    fps: float,
    n_frames: int,
) -> dict[str, Array1D]:
    """Rolling-window mean, std, min, max, kurtosis for each primitive feature.

    Applies the same ``ROLL_WINDOW_DIVISORS`` convention as the generic
    feature battery. All HPW features are aggregate-level signals,
    so every feature gets rolling aggregates (no filtering needed).

    Parameters
    ----------
    features : dict[str, Array1D]
        Primitive feature arrays (R01-R06, W01-W19).
    fps : float
        Frames per second.
    n_frames : int
        Number of frames (to guard against absurd window sizes).

    Returns:
    -------
    dict[str, Array1D]
        Rolling feature arrays keyed as ``{name}_{stat}_w{frames}``.
    """
    roll_windows = sorted(
        {
            w
            for d in ROLL_WINDOW_DIVISORS
            if (w := max(2, int(fps / d))) <= n_frames / 2
        }
    )

    aggs: dict[str, Array1D] = {}
    for wf in roll_windows:
        for key, arr in features.items():
            stats = _rolling_window_stats(arr, wf)
            for stat_name in ("_mean", "_std", "_min", "_max", "_kurt"):
                aggs[f"{key}{stat_name}_w{wf}"] = _ffill_bfill_1d(stats[stat_name])

    return aggs


def _compute_cross_features(
    features: dict[str, Array1D],
    body_stillness_frames: int,
) -> dict[str, Array1D]:
    """Cross-feature aggregations that combine multiple primitive signals.

    Computes features that require interaction between primitives — the
    classifier cannot derive these from individual rolling stats alone.

    Parameters
    ----------
    features : dict[str, Array1D]
        Primitive feature arrays. Must contain body velocity (W15),
        paw elevations (W04, W10), and paw vertical velocities (W01, W07).
    body_stillness_frames : int
        Window size in frames for the body stillness computation.

    Returns:
    -------
    dict[str, Array1D]
        Cross-feature arrays.
    """
    f: dict[str, Array1D] = {}

    # ── X01: body stillness — rolling std of hind body vertical velocity ──
    w15 = features[W15_HIND_BODY_VERTICAL_V_MM_S]
    w15_mean = _smooth_uniform(w15, body_stillness_frames)
    w15_mean_sq = _smooth_uniform(np.square(w15), body_stillness_frames)
    w15_var = np.maximum(w15_mean_sq - np.square(w15_mean), 0.0)
    f[X01_BODY_STILLNESS_MM_S] = _ffill_bfill_1d(np.sqrt(w15_var))

    # ── X02/X03: paw lift ratio — elevation vs velocity smooth ──
    # High ratio = paw elevated but slow → holding paw up (withdrawal)
    # Low ratio  = paw moving fast relative to height → stepping/walking
    for elev_key, vy_key, cross_key in [
        (W04_R_PAW_ELEVATION_MM, W01_R_PAW_VERTICAL_V_MM_S, X02_R_PAW_LIFT_RATIO),
        (W10_L_PAW_ELEVATION_MM, W07_L_PAW_VERTICAL_V_MM_S, X03_L_PAW_LIFT_RATIO),
    ]:
        elev_smooth = _smooth_uniform(features[elev_key], body_stillness_frames)
        vy_smooth = _smooth_uniform(features[vy_key], body_stillness_frames)
        f[cross_key] = _ffill_bfill_1d(
            np.divide(
                elev_smooth,
                np.abs(vy_smooth) + _EPS,
                out=np.zeros_like(elev_smooth, dtype=np.float64),
            )
        )

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# Rearing feature computation
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_rearing_features(
    xy: dict[str, tuple[Array1D, Array1D]],
    floor_y: Array1D,
    px_per_mm: float,
    fps: float,
) -> dict[str, Array1D]:
    """Compute all rearing features.

    Parameters
    ----------
    xy : dict
        {bodypart: (x_array, y_array)} for all bodyparts.
    floor_y : Array1D
        Estimated floor y-position (image coords).
    px_per_mm : float
        Pixels per mm scale.
    fps : float
        Frames per second.

    Returns:
    -------
    dict[str, Array1D]
        Rearing feature arrays.
    """
    f: dict[str, Array1D] = {}

    # ── R01: back angle from horizontal ──
    mb_x, mb_y = xy[MID_BACK]
    lb_x, lb_y = xy[LOWER_BACK]
    dx = lb_x - mb_x
    dy = lb_y - mb_y
    f[R01_BACK_ANGLE_DEG] = _angle_from_horizontal_deg(dx, dy)

    # ── R02: nose elevation above floor ──
    nose_x, nose_y = xy[NOSE]
    f[R02_NOSE_ELEVATION_MM] = (floor_y - nose_y) / px_per_mm

    # ── R03: body elongation ratio ──
    body_vertical = floor_y - nose_y  # nose above floor
    body_horizontal = np.abs(
        nose_x - np.mean([xy[TAIL_BASE][0], xy[TAIL_TIP][0]], axis=0)
    )
    f[R03_BODY_ELONGATION_RATIO] = np.divide(
        body_vertical,
        body_horizontal,
        out=np.zeros_like(body_vertical, dtype=np.float64),
        where=body_horizontal > _EPS,
    )

    # ── R04: centroid vertical velocity ──
    all_y = [xy[bp][1] for bp in ALL_BODYPARTS]
    centroid_y = np.mean(np.column_stack(all_y), axis=1)
    f[R04_CENTROID_VERTICAL_VELOCITY_MM_S] = _vertical_velocity(
        centroid_y, px_per_mm, fps
    )

    # ── R05: front paw elevation relative to hind paws ──
    front_toe_mean_y = np.mean([xy[FRONT_TOE_R][1], xy[FRONT_TOE_L][1]], axis=0)
    hind_toe_mean_y = np.mean([xy[HIND_TOE_R][1], xy[HIND_TOE_L][1]], axis=0)
    f[R05_FRONT_PAW_ELEVATION_MM] = (hind_toe_mean_y - front_toe_mean_y) / px_per_mm

    # ── R06: nose vertical velocity ──
    f[R06_NOSE_VERTICAL_VELOCITY_MM_S] = _vertical_velocity(nose_y, px_per_mm, fps)

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# Hind paw withdrawal feature computation
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_withdrawal_features(
    xy: dict[str, tuple[Array1D, Array1D]],
    floor_y: Array1D,
    px_per_mm: float,
    fps: float,
) -> dict[str, Array1D]:
    """Compute all hind paw withdrawal features.

    Parameters
    ----------
    xy : dict
        {bodypart: (x_array, y_array)} for all bodyparts.
    floor_y : Array1D
        Estimated floor y-position (image coords).
    px_per_mm : float
        Pixels per mm scale.
    fps : float
        Frames per second.

    Returns:
    -------
    dict[str, Array1D]
        Withdrawal feature arrays.
    """
    f: dict[str, Array1D] = {}

    # ── Per-paw kinematics ──
    paws: list[tuple[str, str, str, str, str, str, str, str, str, str, str, str]] = [
        (
            "r",
            HIND_TOE_R,
            HIND_HEEL_R,
            HIND_KNEE_R,
            W01_R_PAW_VERTICAL_V_MM_S,
            W02_R_PAW_HORIZONTAL_V_MM_S,
            W03_R_PAW_VY_TO_VX_RATIO,
            W04_R_PAW_ELEVATION_MM,
            W05_R_PAW_HEEL_TOE_DIST_MM,
            W06_R_PAW_VERTICAL_A_MM_S2,
            W18_R_PAW_VY_PEAK_MM_S,
        ),
        (
            "l",
            HIND_TOE_L,
            HIND_HEEL_L,
            HIND_KNEE_L,
            W07_L_PAW_VERTICAL_V_MM_S,
            W08_L_PAW_HORIZONTAL_V_MM_S,
            W09_L_PAW_VY_TO_VX_RATIO,
            W10_L_PAW_ELEVATION_MM,
            W11_L_PAW_HEEL_TOE_DIST_MM,
            W12_L_PAW_VERTICAL_A_MM_S2,
            W19_L_PAW_VY_PEAK_MM_S,
        ),
    ]

    paw_vy: dict[str, Array1D] = {}
    paw_vx: dict[str, Array1D] = {}

    for (
        side,
        toe_bp,
        heel_bp,
        _knee_bp,
        vy_key,
        vx_key,
        ratio_key,
        elev_key,
        ht_key,
        a_key,
        peak_key,
    ) in paws:
        toe_x, toe_y = xy[toe_bp]
        heel_x, heel_y = xy[heel_bp]

        # Use toe for velocity (most distal, best signal for lift)
        vy = _vertical_velocity(toe_y, px_per_mm, fps)
        vx = _horizontal_velocity(toe_x, px_per_mm, fps)
        paw_vy[side] = vy
        paw_vx[side] = vx

        f[vy_key] = vy
        f[vx_key] = vx

        # vy/vx ratio: high = withdrawal (up without forward), low = stepping
        f[ratio_key] = np.divide(
            np.abs(vy),
            np.abs(vx) + _EPS,
            out=np.zeros_like(vy, dtype=np.float64),
        )

        # Elevation above floor
        f[elev_key] = (floor_y - toe_y) / px_per_mm

        # Heel-toe distance (paw extension/withdrawal)
        f[ht_key] = _euclidean(toe_x, toe_y, heel_x, heel_y, px_per_mm)

        # Vertical acceleration
        f[a_key] = _acceleration_from_velocity(vy, fps)

        # Local vy peak envelope
        peak_win = max(1, int(fps * _VY_PEAK_WINDOW_SEC))
        f[peak_key] = _local_peak(np.abs(vy), peak_win)

    # ── Asymmetry features ──
    r_elev = f[W04_R_PAW_ELEVATION_MM]
    l_elev = f[W10_L_PAW_ELEVATION_MM]
    f[W13_PAW_ELEVATION_ASYMMETRY_MM] = np.abs(r_elev - l_elev)

    r_vy = paw_vy["r"]
    l_vy = paw_vy["l"]
    f[W14_PAW_VERTICAL_V_ASYMMETRY_MM_S] = np.abs(r_vy - l_vy)

    # ── Hind body vertical velocity (control signal: what the body is doing) ──
    body_y_arrays = [xy[bp][1] for bp in [MID_BACK, LOWER_BACK, TAIL_BASE, TAIL_TIP]]
    hind_body_y = np.mean(np.column_stack(body_y_arrays), axis=1)
    f[W15_HIND_BODY_VERTICAL_V_MM_S] = _vertical_velocity(hind_body_y, px_per_mm, fps)

    # ── Paw velocity relative to body (paw minus body = isolated paw movement) ──
    body_vy = f[W15_HIND_BODY_VERTICAL_V_MM_S]
    f[W16_R_PAW_RELATIVE_VERTICAL_V_MM_S] = r_vy - body_vy
    f[W17_L_PAW_RELATIVE_VERTICAL_V_MM_S] = l_vy - body_vy

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════


def compute_hpw_features(
    keypoints_df: pl.DataFrame,
    fps: float,
    px_per_mm: float,
) -> pl.DataFrame:
    """Compute hind paw withdrawal and rearing features from keypoints.

    Parameters
    ----------
    keypoints_df : pl.DataFrame
        Long-form KEYPOINTS_SCHEMA DataFrame for a single experiment.
        Expected to contain ``individual == "rat"`` with 17 animal bodyparts
        (plus optional ``arena_r``, ``arena_l`` static markers).
    fps : float
        Frames per second.
    px_per_mm : float
        Pixels per mm scale factor.

    Returns:
    -------
    pl.DataFrame
        Wide features DataFrame with ``frame`` column + all feature columns.
    """
    xy = _get_bodypart_xy_dict(keypoints_df, ALL_BODYPARTS)

    floor_y = _estimate_floor_y(
        [xy[bp][1] for bp in HIND_PAW_ALL_BPTS],
        fps,
    )

    features: dict[str, Array1D] = {}
    features |= _compute_rearing_features(xy, floor_y, px_per_mm, fps)
    features |= _compute_withdrawal_features(xy, floor_y, px_per_mm, fps)

    n_frames = features[next(iter(features))].shape[0]

    features |= _compute_rolling_aggregates(features, fps, n_frames)
    features |= _compute_cross_features(
        features,
        body_stillness_frames=max(2, int(fps / 5)),
    )

    frames = keypoints_df.select("frame").unique().sort("frame").to_series().to_numpy()

    col_data: dict[str, np.ndarray] = {"frame": frames.astype(np.int64)}
    col_data |= {k: v.astype(np.float64) for k, v in features.items()}

    return pl.DataFrame(col_data)
