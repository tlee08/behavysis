"""Functions have the following format."""

import numpy as np
import polars as pl
from pydantic import BaseModel, PositiveFloat

from behavysis.constants import LIKELIHOOD, SINGLE
from behavysis.models import ExperimentConfig, ExperimentMetadata
from behavysis.transforms import check_bpts_exist

# ═══════════════════════════════════════════════════════════════════════════════
# Config Models
# ═══════════════════════════════════════════════════════════════════════════════


class PxPerMmConfig(BaseModel):
    """PxPerMmConfig."""

    pt_a: str
    pt_b: str
    dist_mm: PositiveFloat
    pcutoff: PositiveFloat


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

    pt_low_likelihood_ls = [
        (_pt, np.nanmax(_pt_df[LIKELIHOOD]))
        for _pt, _pt_df in [(pt_a, pt_a_df), (pt_b, pt_b_df)]
        if not np.any(_pt_df[LIKELIHOOD] > pcutoff)
    ]
    if pt_low_likelihood_ls:
        _names = ", ".join([_i[0] for _i in pt_low_likelihood_ls])
        _maxes = ", ".join([f"({_i[1]}: {_i[1]})" for _i in pt_low_likelihood_ls])
        msg = (
            f"No points for: {_names}\n"
            f"pcutoff is {pcutoff}.\n"
            f"Highest likelihoods are: {_maxes}"
        )
        raise ValueError(msg)

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

    metadata.px_per_mm = float(px_per_mm_val)
    return metadata
