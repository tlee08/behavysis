"""Functions have the following format."""

import numpy as np
import polars as pl
from pydantic import BaseModel, PositiveFloat

from behavysis.constants import LIKELIHOOD, SINGLE
from behavysis.models import ExperimentConfig, ExperimentMetadata
from behavysis.schemas import check_bpts_exist

# ═══════════════════════════════════════════════════════════════════════════════
# Config Models
# ═══════════════════════════════════════════════════════════════════════════════


class PxPerMmConfig(BaseModel):
    """PxPerMmConfig."""

    pt_a: str = "pt_a"
    pt_b: str = "pt_b"
    pcutoff: PositiveFloat = 0.5
    dist_mm: PositiveFloat = 400.0


# ═══════════════════════════════════════════════════════════════════════════════
# Functions
# ═══════════════════════════════════════════════════════════════════════════════


def px_per_mm(
    keypoints_df: pl.DataFrame,
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
) -> ExperimentMetadata:
    """Calculates the pixels per mm conversion using calibration points."""
    cfg = config.require_calculate_parameters().require(
        "px_per_mm",
        PxPerMmConfig,
    )
    pt_a = cfg.pt_a
    pt_b = cfg.pt_b
    pcutoff = cfg.pcutoff
    dist_mm = cfg.dist_mm

    check_bpts_exist(keypoints_df, [pt_a, pt_b])

    # Get calibration point coordinates for "single" individual
    pt_a_df = (
        keypoints_df.filter(
            pl.col("individual") == SINGLE,
            pl.col("bodypart") == pt_a,
        )
        .sort("frame")
        .to_pandas()
    )
    pt_b_df = (
        keypoints_df.filter(
            pl.col("individual") == SINGLE,
            pl.col("bodypart") == pt_b,
        )
        .sort("frame")
        .to_pandas()
    )

    for pt_df, pt in [(pt_a_df, pt_a), (pt_b_df, pt_b)]:
        assert np.any(pt_df[LIKELIHOOD] > pcutoff), (
            f'No points for "{pt}" are above the pcutoff of {pcutoff}.\n'
            f"Consider lowering the pcutoff in the config file.\n"
            f'The highest likelihood value in "{pt}" is '
            f"{np.nanmax(pt_df[LIKELIHOOD])}."
        )

    # Interpolate low-likelihood points
    for pt_df in [pt_a_df, pt_b_df]:
        mask = pt_df[LIKELIHOOD] < pcutoff
        pt_df.loc[mask, "x"] = np.nan
        pt_df.loc[mask, "y"] = np.nan
        pt_df["x"] = pt_df["x"].interpolate(method="linear").bfill().ffill()
        pt_df["y"] = pt_df["y"].interpolate(method="linear").bfill().ffill()

    dist_px = np.nanmean(
        np.sqrt(
            np.square(pt_a_df["x"] - pt_b_df["x"])
            + np.square(pt_a_df["y"] - pt_b_df["y"]),
        ),
    )
    px_per_mm_val = dist_px / dist_mm

    metadata.px_per_mm = px_per_mm_val
    return metadata
