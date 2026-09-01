"""Feature extraction for rearing behaviours.

Single rat, side-on camera view, glass cylinder chamber.
Designed for DeepLabCut multi-animal keypoint tracking with 17 bodyparts
(plus 2 static arena markers).

Features are grouped into one behaviour-specific battery:

- **Rearing** (R01-R08): back angle, head elevation, body elongation,
    centroid velocity, front paw elevation, head velocity, whole-body tilt,
    upper-body angle.

A robust bottom reference is computed from likelihood-filtered
hind-paw bodyparts so that rearing posture can be estimated
even when the back/tail are occluded (e.g. rat facing the camera).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from pydantic import BaseModel

from ._helper import (
    _EPS,
    ALL_BODYPARTS,
    ARENA_BPTS,
    ARENA_INDIVIDUAL,
    EAR_L,
    EAR_R,
    FRONT_TOE_L,
    FRONT_TOE_R,
    HIND_TOE_L,
    HIND_TOE_R,
    LOWER_BACK,
    MID_BACK,
    NOSE,
    RAT_INDIVIDUAL,
    _angle_from_horizontal_deg,
    _compute_bottom_reference,
    _compute_rolling_aggregates,
    _estimate_floor_y,
    _get_bodypart_xy_dict,
    _vertical_velocity,
)

if TYPE_CHECKING:
    from behavysis.constants import Array1D
    from behavysis.models import ExperimentConfig, ExperimentMetadata

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════


class ExtractRearingConfig(BaseModel):
    """Configuration for rearing feature extraction."""

    pcutoff: float = 0.6


# ═══════════════════════════════════════════════════════════════════════════════
# Feature name constants — Rearing
# ═══════════════════════════════════════════════════════════════════════════════

R01_BACK_ANGLE_DEG = "R01_back_angle_deg"
R02_HEAD_ELEVATION_MM = "R02_head_elevation_mm"
R03_BODY_ELONGATION_RATIO = "R03_body_elongation_ratio"
R04_CENTROID_VERTICAL_VELOCITY_MM_S = "R04_centroid_vertical_velocity_mm_s"
R05_FRONT_PAW_ELEVATION_MM = "R05_front_paw_elevation_mm"
R06_HEAD_VERTICAL_VELOCITY_MM_S = "R06_head_vertical_velocity_mm_s"
R07_WHOLE_BODY_ANGLE_DEG = "R07_whole_body_angle_deg"
R08_UPPER_BODY_ANGLE_DEG = "R08_upper_body_angle_deg"

REARING_FEATURES: list[str] = [
    R01_BACK_ANGLE_DEG,
    R02_HEAD_ELEVATION_MM,
    R03_BODY_ELONGATION_RATIO,
    R04_CENTROID_VERTICAL_VELOCITY_MM_S,
    R05_FRONT_PAW_ELEVATION_MM,
    R06_HEAD_VERTICAL_VELOCITY_MM_S,
    R07_WHOLE_BODY_ANGLE_DEG,
    R08_UPPER_BODY_ANGLE_DEG,
]

# ═══════════════════════════════════════════════════════════════════════════════
# Rearing feature computation
# ═══════════════════════════════════════════════════════════════════════════════


def _nanmean(arrays: list[Array1D]) -> Array1D:
    """Column-wise mean of stacked 1D arrays, ignoring NaN."""
    with np.errstate(all="ignore"):
        return np.nanmean(np.column_stack(arrays), axis=1)


def _compute_rearing_features(  # noqa: PLR0913
    xy: dict[str, tuple[Array1D, Array1D]],
    floor_y: Array1D,
    bottom_x: Array1D,
    bottom_y: Array1D,
    px_per_mm: float,
    fps: float,
) -> dict[str, Array1D]:
    """Compute all rearing features."""
    f: dict[str, Array1D] = {}

    head_x = _nanmean([xy[NOSE][0], xy[EAR_R][0], xy[EAR_L][0]])
    head_y = _nanmean([xy[NOSE][1], xy[EAR_R][1], xy[EAR_L][1]])

    # -- R01: back angle from horizontal --
    mb_x, mb_y = xy[MID_BACK]
    lb_x, lb_y = xy[LOWER_BACK]
    dx = lb_x - mb_x
    dy = lb_y - mb_y
    f[R01_BACK_ANGLE_DEG] = _angle_from_horizontal_deg(dx, dy)

    # -- R02: head elevation above floor --
    f[R02_HEAD_ELEVATION_MM] = (floor_y - head_y) / px_per_mm

    # -- R03: body elongation ratio --
    body_vertical = bottom_y - head_y
    body_horizontal = np.abs(head_x - bottom_x)
    f[R03_BODY_ELONGATION_RATIO] = np.divide(body_vertical, body_horizontal + _EPS)

    # -- R04: centroid vertical velocity --
    all_y = [xy[bp][1] for bp in ALL_BODYPARTS]
    centroid_y = _nanmean(all_y)
    f[R04_CENTROID_VERTICAL_VELOCITY_MM_S] = _vertical_velocity(
        centroid_y, px_per_mm, fps
    )

    # -- R05: front paw elevation relative to hind paws --
    front_toe_mean_y = _nanmean([xy[FRONT_TOE_R][1], xy[FRONT_TOE_L][1]])
    hind_toe_mean_y = _nanmean([xy[HIND_TOE_R][1], xy[HIND_TOE_L][1]])
    f[R05_FRONT_PAW_ELEVATION_MM] = (hind_toe_mean_y - front_toe_mean_y) / px_per_mm

    # -- R06: head vertical velocity --
    f[R06_HEAD_VERTICAL_VELOCITY_MM_S] = _vertical_velocity(head_y, px_per_mm, fps)

    # -- R07: whole-body angle (bottom->head vector from horizontal) --
    f[R07_WHOLE_BODY_ANGLE_DEG] = _angle_from_horizontal_deg(
        head_x - bottom_x,
        head_y - bottom_y,
    )

    # -- R08: upper-body angle (LOWER_BACK->head vector from horizontal) --
    f[R08_UPPER_BODY_ANGLE_DEG] = _angle_from_horizontal_deg(
        head_x - lb_x,
        head_y - lb_y,
    )

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════


def rearing_compute(
    keypoints_df: pl.DataFrame,
    fps: float,
    px_per_mm: float,
    pcutoff: float = 0.6,
) -> pl.DataFrame:
    """Compute rearing features from keypoints.

    Parameters
    ----------
    keypoints_df : pl.DataFrame
        Long-form KEYPOINTS_SCHEMA DataFrame for a single experiment.
    fps : float
        Frames per second.
    px_per_mm : float
        Pixels per mm scale factor.
    pcutoff : float
        Likelihood threshold for bottom-reference bodypart presence.

    Returns:
    -------
    pl.DataFrame
        Wide features DataFrame with ``frame`` column + all feature columns.
    """
    xy = _get_bodypart_xy_dict(keypoints_df, ALL_BODYPARTS, RAT_INDIVIDUAL, pcutoff)

    arena_xy = _get_bodypart_xy_dict(keypoints_df, ARENA_BPTS, ARENA_INDIVIDUAL)
    floor_y = _estimate_floor_y(arena_xy, xy, fps)

    bottom_x, bottom_y = _compute_bottom_reference(keypoints_df, pcutoff)

    features: dict[str, Array1D] = {}
    features |= _compute_rearing_features(
        xy,
        floor_y,
        bottom_x,
        bottom_y,
        px_per_mm,
        fps,
    )

    features |= _compute_rolling_aggregates(features, fps)

    frames = keypoints_df.select("frame").unique().sort("frame").to_series().to_numpy()

    col_data: dict[str, np.ndarray] = {"frame": frames.astype(np.int64)}
    col_data |= {k: v.astype(np.float64) for k, v in features.items()}

    return pl.DataFrame(col_data)


def extract_rearing(
    keypoints_df: pl.DataFrame,
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
) -> pl.DataFrame:
    """Protocol-compliant wrapper for rearing feature extraction."""
    cfg = config.require_extract_features().require(
        "extract_rearing", ExtractRearingConfig
    )
    return rearing_compute(
        keypoints_df,
        fps=metadata.require_fps(),
        px_per_mm=metadata.require_px_per_mm(),
        pcutoff=cfg.pcutoff,
    )
