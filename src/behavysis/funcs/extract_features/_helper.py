"""Shared constants, helpers, and infrastructure for feature extraction.

Extractor protocol, bodypart constants, kinematics helpers, rolling-window
aggregation, floor estimation, and likelihood-weighted bottom-reference
computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
import polars as pl
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

if TYPE_CHECKING:
    from behavysis.constants import Array1D, Array2D
    from behavysis.models import ExperimentConfig, ExperimentMetadata


class ExtractFeaturesFunc(Protocol):
    """Protocol for extract features functions."""

    __name__: str

    def __call__(
        self,
        keypoints_df: pl.DataFrame,
        config: ExperimentConfig,
        metadata: ExperimentMetadata,
    ) -> pl.DataFrame:
        """Protocol for extract features functions."""


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

RAT_INDIVIDUAL = "rat"
ARENA_INDIVIDUAL = "single"

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
BOTTOM_BPTS = [LOWER_BACK, TAIL_BASE]

# ═══════════════════════════════════════════════════════════════════════════════
# Smoothing / derivative parameters
# ═══════════════════════════════════════════════════════════════════════════════

_POS_SMOOTH_WINDOW: int = 3
_VEL_SMOOTH_WINDOW: int = 3
_FLOOR_ROLL_WINDOW_SEC: float = 5.0
_VY_PEAK_WINDOW_SEC: float = 0.2
_EPS: float = 1e-6

# ═══════════════════════════════════════════════════════════════════════════════
# Rolling window parameters
# ═══════════════════════════════════════════════════════════════════════════════

ROLL_WINDOW_SECONDS: list[float] = [1.0, 0.5, 0.25, 0.2, 0.1]


def _rolling_windows(
    roll_window_seconds: list[float],
    fps: float,
    n_frames: int,
) -> list[tuple[str, int]]:
    """Window ``(label, frames)`` pairs that fit the video.

    Label is the window duration in seconds (fps-agnostic);
    ``frames`` is the nearest integer frame count for the given fps.
    """
    return [
        (f"{s:g}s", max(2, round(fps * s)))
        for s in roll_window_seconds
        if max(2, round(fps * s)) <= n_frames / 2
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers — DataFrame → numpy extraction
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_bodypart_xy(
    keypoints_df: pl.DataFrame,
    bodypart: str,
    individual: str,
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


def _extract_bodypart_xyl(
    keypoints_df: pl.DataFrame,
    bodypart: str,
    individual: str,
) -> tuple[Array1D, Array1D, Array1D, int]:
    """Extract (x, y, likelihood) with NaN for missing frames — no fill."""
    bp_df = (
        keypoints_df.filter(
            pl.col("individual") == individual,
            pl.col("bodypart") == bodypart,
        )
        .select(["frame", "x", "y", "likelihood"])
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
    bp_lik = bp_df.select("likelihood").to_series().to_numpy()

    x_arr = np.full(n_frames, np.nan, dtype=np.float64)
    y_arr = np.full(n_frames, np.nan, dtype=np.float64)
    lik_arr = np.full(n_frames, np.nan, dtype=np.float64)
    for i, f in enumerate(bp_frames):
        idx = frame_to_idx[int(f)]
        x_arr[idx] = bp_x[i]
        y_arr[idx] = bp_y[i]
        lik_arr[idx] = bp_lik[i]

    return x_arr, y_arr, lik_arr, n_frames


def _get_bodypart_xy_dict(
    keypoints_df: pl.DataFrame,
    bodyparts: list[str],
    individual: str,
    pcutoff: float | None = None,
) -> dict[str, tuple[Array1D, Array1D]]:
    """Return ``{bodypart: (x, y)}`` for a list of bodyparts.

    When ``pcutoff`` is given, positions whose likelihood is below
    ``pcutoff`` (or missing) are set to NaN so they propagate through the
    feature pipeline instead of being silently forward/backward filled.
    """
    result: dict[str, tuple[Array1D, Array1D]] = {}
    for bp in bodyparts:
        if pcutoff is None:
            x_arr, y_arr, _ = _extract_bodypart_xy(keypoints_df, bp, individual)
        else:
            x_arr, y_arr, lik_arr, _ = _extract_bodypart_xyl(
                keypoints_df, bp, individual
            )
            mask = (lik_arr < pcutoff) | ~np.isfinite(lik_arr)
            x_arr = np.where(mask, np.nan, x_arr)
            y_arr = np.where(mask, np.nan, y_arr)
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
    """Apply uniform (boxcar) smoothing to a 1D array.

    NaN-aware: NaN values are ignored within each window, and an all-NaN
    window yields NaN.  Identical to ``uniform_filter1d`` for NaN-free input.
    """
    if window <= 1:
        return arr.copy()
    arr = np.asarray(arr, dtype=np.float64)
    mask = np.isfinite(arr)
    if mask.all():
        return uniform_filter1d(arr, size=window, mode="nearest")
    a0 = np.where(mask, arr, 0.0)
    cnt = uniform_filter1d(mask.astype(np.float64), size=window, mode="nearest")
    out = uniform_filter1d(a0, size=window, mode="nearest") / np.maximum(cnt, _EPS)
    return np.where(cnt > 0, out, np.nan)


def _vertical_velocity(
    y: Array1D,
    px_per_mm: float,
    fps: float,
    smooth_window: int = _POS_SMOOTH_WINDOW,
) -> Array1D:
    """Upward velocity in mm/s from y-coordinate array.

    Positive = upward movement (y decreases in image space -> we negate diff).
    First frame is padded with 0 (no movement before start).
    """
    y_smooth = _smooth_uniform(y, smooth_window)
    vy_raw = -np.diff(y_smooth, prepend=y_smooth[0])
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
    """Acceleration from velocity array, in units/s^2."""
    v_smooth = _smooth_uniform(v, smooth_window)
    a_raw = np.diff(v_smooth, prepend=v_smooth[0])
    return a_raw * fps


def _local_peak(
    signal: Array1D,
    window_frames: int,
) -> Array1D:
    """Per-frame local maximum of signal within +/-window_frames.

    Produces a smoothed envelope that captures local peak intensity
    of the signal -- useful for detecting transient spikes (e.g. paw flinch).
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
    arena_xy: dict[str, tuple[Array1D, Array1D]],
    xy: dict[str, tuple[Array1D, Array1D]],
    fps: float,
) -> Array1D:
    """Estimate floor y-position from static arena markers.

    Uses arena_r and arena_l (glass plate edge markers) as the primary
    floor reference.  Falls back to a rolling 90th-percentile estimate
    from BOTTOM_BPTS if arena markers are unavailable.
    """
    arena_y = np.mean([arena_xy[ARENA_R][1], arena_xy[ARENA_L][1]], axis=0)

    if np.isfinite(arena_y).any():
        floor_y_val = np.median(arena_y[np.isfinite(arena_y)])
        return np.full_like(arena_y, floor_y_val)

    bottom_y = [xy[bp][1] for bp in BOTTOM_BPTS]
    stacked = np.column_stack(bottom_y)
    floor_raw = np.percentile(stacked, 90, axis=1)
    roll_frames = max(1, min(int(fps * _FLOOR_ROLL_WINDOW_SEC), len(floor_raw)))
    floor = _smooth_uniform(floor_raw, roll_frames)
    floor[:roll_frames] = floor[roll_frames]
    return floor


# ═══════════════════════════════════════════════════════════════════════════════
# Likelihood-weighted bottom reference
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_bottom_reference(
    keypoints_df: pl.DataFrame,
    pcutoff: float,
) -> tuple[Array1D, Array1D]:
    """Hind-paw bottom reference (x, y), NaN where all paws are occluded.

    Averages (x, y) of the hind-paw bodyparts whose likelihood exceeds
    ``pcutoff``.  During rearing the hind paws stay planted on the floor,
    so they anchor the true base of the animal (the back/tail rise and are
    deliberately excluded).  No forward-fill: an all-occluded frame yields
    NaN so the occlusion propagates through the feature pipeline.
    """
    n_frames = keypoints_df.select("frame").unique().sort("frame").to_series().len()

    bottom_x = np.full(n_frames, np.nan, dtype=np.float64)
    bottom_y = np.full(n_frames, np.nan, dtype=np.float64)

    count = np.zeros(n_frames, dtype=np.int32)
    for bp in HIND_PAW_ALL_BPTS:
        x_arr, y_arr, lik_arr, _ = _extract_bodypart_xyl(
            keypoints_df, bp, RAT_INDIVIDUAL
        )
        valid = (lik_arr > pcutoff) & np.isfinite(x_arr) & np.isfinite(y_arr)
        bottom_x[valid] = np.nansum(
            np.column_stack([bottom_x[valid], x_arr[valid]]), axis=1
        )
        bottom_y[valid] = np.nansum(
            np.column_stack([bottom_y[valid], y_arr[valid]]), axis=1
        )
        count[valid] += 1

    present = count > 0
    if present.any():
        bottom_x[present] /= count[present].astype(np.float64)
        bottom_y[present] /= count[present].astype(np.float64)

    return bottom_x, bottom_y


# ═══════════════════════════════════════════════════════════════════════════════
# Rolling window aggregation
# ═══════════════════════════════════════════════════════════════════════════════


def _rolling_window_stats(
    arr: Array1D,
    window: int,
) -> dict[str, Array1D]:
    """NaN-aware rolling mean, std, min, max.

    Centred windows with ``mode="nearest"`` edge handling, matching the
    scipy filters for NaN-free input.  NaN values are ignored within each
    window; an all-NaN window yields NaN.

    Returns ``{"_mean", "_std", "_min", "_max"}`` arrays.
    """
    arr = np.asarray(arr, dtype=np.float64)
    mask = np.isfinite(arr)
    cnt = uniform_filter1d(mask.astype(np.float64), size=window, mode="nearest")
    a0 = np.where(mask, arr, 0.0)
    m = uniform_filter1d(a0, size=window, mode="nearest") / np.maximum(cnt, _EPS)
    m2 = uniform_filter1d(
        np.where(mask, arr * arr, 0.0), size=window, mode="nearest"
    ) / np.maximum(cnt, _EPS)
    var = np.maximum(m2 - np.square(m), 0.0)
    std = np.sqrt(var)
    lo = minimum_filter1d(np.where(mask, arr, np.inf), size=window, mode="nearest")
    hi = maximum_filter1d(np.where(mask, arr, -np.inf), size=window, mode="nearest")
    valid = cnt > 0
    return {
        "_mean": np.where(valid, m, np.nan),
        "_std": np.where(valid, std, np.nan),
        "_min": np.where(valid, lo, np.nan),
        "_max": np.where(valid, hi, np.nan),
    }


def _compute_rolling_aggregates(
    features: dict[str, Array1D],
    fps: float,
    n_frames: int,
) -> dict[str, Array1D]:
    """Rolling-window mean, std, min, max for each primitive feature.

    Applies the same ``ROLL_WINDOW_SECONDS`` convention as the generic
    feature battery. All features are aggregate-level signals,
    so every feature gets rolling aggregates (no filtering needed).

    Returns rolling feature arrays keyed as ``{name}_{stat}_w{seconds}``.
    """
    aggs: dict[str, Array1D] = {}
    for label, wf in _rolling_windows(ROLL_WINDOW_SECONDS, fps, n_frames):
        for key, arr in features.items():
            stats = _rolling_window_stats(arr, wf)
            for stat_name in ("_mean", "_std", "_min", "_max"):
                aggs[f"{key}{stat_name}_w{label}"] = stats[stat_name]

    return aggs
