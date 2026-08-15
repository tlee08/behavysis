"""Feature extraction for hind paw withdrawal behaviours.

Single rat, side-on camera view, glass cylinder chamber.
Designed for DeepLabCut multi-animal keypoint tracking with 17 bodyparts
(plus 2 static arena markers).

Features are grouped into one behaviour-specific battery:

- **Hind Paw Withdrawal** (W01-W29): paw vertical/horizontal kinematics,
    elevation above estimated floor, heel-toe distance, knee posture,
    paw velocity direction, paw area, asymmetry, and body-stillness
    control features that help distinguish isolated paw lifts from
    stepping/walking.

Coordinate convention (image space):
    x -> rightward positive, y -> downward positive.
    "Upward" movement in the world = y *decreases* in image space.
    Features report upward velocity/acceleration as *positive* values
    (i.e. we negate the raw image-space derivatives).

Cross features (X01-X05): body stillness, paw lift ratio, and hold score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from pydantic import BaseModel

from ._helper import (
    _EPS,
    _VY_PEAK_WINDOW_SEC,
    ALL_BODYPARTS,
    ARENA_BPTS,
    ARENA_INDIVIDUAL,
    HIND_HEEL_L,
    HIND_HEEL_R,
    HIND_KNEE_L,
    HIND_KNEE_R,
    HIND_TOE_L,
    HIND_TOE_R,
    LOWER_BACK,
    MID_BACK,
    RAT_INDIVIDUAL,
    TAIL_BASE,
    TAIL_TIP,
    _acceleration_from_velocity,
    _angle_from_horizontal_deg,
    _compute_rolling_aggregates,
    _estimate_floor_y,
    _euclidean,
    _ffill_bfill_1d,
    _get_bodypart_xy_dict,
    _horizontal_velocity,
    _local_peak,
    _smooth_uniform,
    _vertical_velocity,
)

if TYPE_CHECKING:
    from behavysis.constants import Array1D
    from behavysis.models import ExperimentConfig, ExperimentMetadata

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

_HOLD_VY_SCALE_MM_S: float = 100.0  # mm/s — vy above this suppresses hold score


class ExtractHpwConfig(BaseModel):
    """Configuration for hind paw withdrawal feature extraction."""

    pcutoff: float = 0.6


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

# --- Right paw extended ---
W20_R_KNEE_ELEVATION_MM = "W20_r_knee_elevation_mm"
W21_R_KNEE_TOE_ANGLE_DEG = "W21_r_knee_toe_angle_deg"
W22_R_KNEE_HIP_DIST_MM = "W22_r_knee_hip_dist_mm"
W23_R_PAW_VELOCITY_DIRECTION_DEG = "W23_r_paw_velocity_direction_deg"
W24_R_PAW_AREA_MM2 = "W24_r_paw_area_mm2"

# --- Left paw extended ---
W25_L_KNEE_ELEVATION_MM = "W25_l_knee_elevation_mm"
W26_L_KNEE_TOE_ANGLE_DEG = "W26_l_knee_toe_angle_deg"
W27_L_KNEE_HIP_DIST_MM = "W27_l_knee_hip_dist_mm"
W28_L_PAW_VELOCITY_DIRECTION_DEG = "W28_l_paw_velocity_direction_deg"
W29_L_PAW_AREA_MM2 = "W29_l_paw_area_mm2"

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
    W20_R_KNEE_ELEVATION_MM,
    W21_R_KNEE_TOE_ANGLE_DEG,
    W22_R_KNEE_HIP_DIST_MM,
    W23_R_PAW_VELOCITY_DIRECTION_DEG,
    W24_R_PAW_AREA_MM2,
    W25_L_KNEE_ELEVATION_MM,
    W26_L_KNEE_TOE_ANGLE_DEG,
    W27_L_KNEE_HIP_DIST_MM,
    W28_L_PAW_VELOCITY_DIRECTION_DEG,
    W29_L_PAW_AREA_MM2,
]

# ═══════════════════════════════════════════════════════════════════════════════
# Cross-feature name constants
# ═══════════════════════════════════════════════════════════════════════════════

X01_BODY_STILLNESS_MM_S = "X01_body_stillness_mm_s"
X02_R_PAW_LIFT_RATIO = "X02_r_paw_lift_ratio"
X03_L_PAW_LIFT_RATIO = "X03_l_paw_lift_ratio"
X04_R_PAW_HOLD_SCORE_MM = "X04_r_paw_hold_score_mm"
X05_L_PAW_HOLD_SCORE_MM = "X05_l_paw_hold_score_mm"

CROSS_FEATURES: list[str] = [
    X01_BODY_STILLNESS_MM_S,
    X02_R_PAW_LIFT_RATIO,
    X03_L_PAW_LIFT_RATIO,
    X04_R_PAW_HOLD_SCORE_MM,
    X05_L_PAW_HOLD_SCORE_MM,
]

# ═══════════════════════════════════════════════════════════════════════════════
# Cross-feature computation
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_cross_features(
    features: dict[str, Array1D],
    body_stillness_frames: int,
) -> dict[str, Array1D]:
    """Cross-feature aggregations that combine multiple primitive signals.

    Computes features that require interaction between primitives -- the
    classifier cannot derive these from individual rolling stats alone.
    """
    f: dict[str, Array1D] = {}

    # -- X01: body stillness -- rolling std of hind body vertical velocity --
    w15 = features[W15_HIND_BODY_VERTICAL_V_MM_S]
    w15_mean = _smooth_uniform(w15, body_stillness_frames)
    w15_mean_sq = _smooth_uniform(np.square(w15), body_stillness_frames)
    w15_var = np.maximum(w15_mean_sq - np.square(w15_mean), 0.0)
    f[X01_BODY_STILLNESS_MM_S] = _ffill_bfill_1d(np.sqrt(w15_var))

    # -- X02/X03: paw lift ratio -- elevation vs velocity smooth --
    # High ratio = paw elevated but slow -> holding paw up (withdrawal)
    # Low ratio  = paw moving fast relative to height -> stepping/walking
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

    # -- X04/X05: hold score -- elevation weighted by velocity decay --
    # High = paw elevated AND slow (holding); Low = paw moving (withdrawal/stepping)
    for elev_key, vy_key, cross_key in [
        (W04_R_PAW_ELEVATION_MM, W01_R_PAW_VERTICAL_V_MM_S, X04_R_PAW_HOLD_SCORE_MM),
        (W10_L_PAW_ELEVATION_MM, W07_L_PAW_VERTICAL_V_MM_S, X05_L_PAW_HOLD_SCORE_MM),
    ]:
        elev_smooth = _smooth_uniform(features[elev_key], body_stillness_frames)
        vy_smooth = _smooth_uniform(features[vy_key], body_stillness_frames)
        f[cross_key] = _ffill_bfill_1d(
            elev_smooth * np.exp(-np.abs(vy_smooth) / _HOLD_VY_SCALE_MM_S)
        )

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
    """Compute all hind paw withdrawal features."""
    f: dict[str, Array1D] = {}

    # -- Per-paw kinematics --
    paws = [
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
            W20_R_KNEE_ELEVATION_MM,
            W21_R_KNEE_TOE_ANGLE_DEG,
            W22_R_KNEE_HIP_DIST_MM,
            W23_R_PAW_VELOCITY_DIRECTION_DEG,
            W24_R_PAW_AREA_MM2,
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
            W25_L_KNEE_ELEVATION_MM,
            W26_L_KNEE_TOE_ANGLE_DEG,
            W27_L_KNEE_HIP_DIST_MM,
            W28_L_PAW_VELOCITY_DIRECTION_DEG,
            W29_L_PAW_AREA_MM2,
        ),
    ]

    paw_vy: dict[str, Array1D] = {}
    paw_vx: dict[str, Array1D] = {}

    for (
        side,
        toe_bp,
        heel_bp,
        knee_bp,
        vy_key,
        vx_key,
        ratio_key,
        elev_key,
        ht_key,
        a_key,
        peak_key,
        knee_elev_key,
        knee_toe_angle_key,
        knee_hip_dist_key,
        vel_dir_key,
        paw_area_key,
    ) in paws:
        toe_x, toe_y = xy[toe_bp]
        heel_x, heel_y = xy[heel_bp]
        knee_x, knee_y = xy[knee_bp]

        vy = _vertical_velocity(toe_y, px_per_mm, fps)
        vx = _horizontal_velocity(toe_x, px_per_mm, fps)
        paw_vy[side] = vy
        paw_vx[side] = vx

        f[vy_key] = vy
        f[vx_key] = vx

        f[ratio_key] = np.divide(
            np.abs(vy),
            np.abs(vx) + _EPS,
            out=np.zeros_like(vy, dtype=np.float64),
        )

        f[elev_key] = (floor_y - toe_y) / px_per_mm

        f[ht_key] = _euclidean(toe_x, toe_y, heel_x, heel_y, px_per_mm)

        f[a_key] = _acceleration_from_velocity(vy, fps)

        peak_win = max(1, int(fps * _VY_PEAK_WINDOW_SEC))
        f[peak_key] = _local_peak(np.abs(vy), peak_win)

        f[knee_elev_key] = (floor_y - knee_y) / px_per_mm

        f[knee_toe_angle_key] = _angle_from_horizontal_deg(
            toe_x - knee_x,
            toe_y - knee_y,
        )

        lb_x, lb_y = xy[LOWER_BACK]
        f[knee_hip_dist_key] = _euclidean(knee_x, knee_y, lb_x, lb_y, px_per_mm)

        f[vel_dir_key] = np.degrees(np.arctan2(vy, np.abs(vx) + _EPS))

        cross = (toe_x - heel_x) * (knee_y - heel_y) - (toe_y - heel_y) * (
            knee_x - heel_x
        )
        f[paw_area_key] = 0.5 * np.abs(cross) / (px_per_mm * px_per_mm)

    # -- Asymmetry features --
    r_elev = f[W04_R_PAW_ELEVATION_MM]
    l_elev = f[W10_L_PAW_ELEVATION_MM]
    f[W13_PAW_ELEVATION_ASYMMETRY_MM] = np.abs(r_elev - l_elev)

    r_vy = paw_vy["r"]
    l_vy = paw_vy["l"]
    f[W14_PAW_VERTICAL_V_ASYMMETRY_MM_S] = np.abs(r_vy - l_vy)

    # -- Hind body vertical velocity (control signal: what the body is doing) --
    body_y_arrays = [xy[bp][1] for bp in [MID_BACK, LOWER_BACK, TAIL_BASE, TAIL_TIP]]
    hind_body_y = np.mean(np.column_stack(body_y_arrays), axis=1)
    f[W15_HIND_BODY_VERTICAL_V_MM_S] = _vertical_velocity(hind_body_y, px_per_mm, fps)

    # -- Paw velocity relative to body (paw minus body = isolated paw movement) --
    body_vy = f[W15_HIND_BODY_VERTICAL_V_MM_S]
    f[W16_R_PAW_RELATIVE_VERTICAL_V_MM_S] = r_vy - body_vy
    f[W17_L_PAW_RELATIVE_VERTICAL_V_MM_S] = l_vy - body_vy

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════


def hpw_compute(
    keypoints_df: pl.DataFrame,
    fps: float,
    px_per_mm: float,
) -> pl.DataFrame:
    """Compute hind paw withdrawal features from keypoints.

    Parameters
    ----------
    keypoints_df : pl.DataFrame
        Long-form KEYPOINTS_SCHEMA DataFrame for a single experiment.
    fps : float
        Frames per second.
    px_per_mm : float
        Pixels per mm scale factor.

    Returns:
    -------
    pl.DataFrame
        Wide features DataFrame with ``frame`` column + all feature columns.
    """
    xy = _get_bodypart_xy_dict(keypoints_df, ALL_BODYPARTS, RAT_INDIVIDUAL)

    arena_xy = _get_bodypart_xy_dict(keypoints_df, ARENA_BPTS, ARENA_INDIVIDUAL)
    floor_y = _estimate_floor_y(arena_xy, xy, fps)

    features: dict[str, Array1D] = {}
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


def extract_hpw(
    keypoints_df: pl.DataFrame,
    config: ExperimentConfig,  # noqa: ARG001
    metadata: ExperimentMetadata,
) -> pl.DataFrame:
    """Protocol-compliant wrapper for HPW feature extraction."""
    return hpw_compute(
        keypoints_df,
        fps=metadata.require_fps(),
        px_per_mm=metadata.require_px_per_mm(),
    )
