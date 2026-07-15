"""Functions have the following format."""

import polars as pl
from pydantic import BaseModel, PositiveFloat

from behavysis.models import ExperimentConfig, ExperimentMetadata

# ═══════════════════════════════════════════════════════════════════════════════
# Config Models
# ═══════════════════════════════════════════════════════════════════════════════


class InterpolateConfig(BaseModel):
    """InterpolateConfig."""

    pcutoff: PositiveFloat = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Functions
# ═══════════════════════════════════════════════════════════════════════════════


def interpolate(
    keypoints_df: pl.DataFrame,
    config: ExperimentConfig,
    metadata: ExperimentMetadata,  # noqa: ARG001
) -> pl.DataFrame:
    """Smooths noticeable jitter of points.

    Where the likelihood (and accuracy) of
    a point's coordinates are low
    (e.g., when the subject's head goes out of view).
    It does this by linearly interpolating the frames
    of a body part that are below a given likelihood pcutoff.
    """
    cfg = config.require_preprocess().require("interpolate", InterpolateConfig)

    # Fill any null likelihoods with 0
    keypoints_df = keypoints_df.with_columns(
        pl.col("likelihood").fill_null(0),
    )

    # Set x and y to null where likelihood is below pcutoff
    keypoints_df = keypoints_df.with_columns(
        pl.when(pl.col("likelihood") >= cfg.pcutoff)
        .then(pl.col("x"))
        .otherwise(None)
        .alias("x"),
        pl.when(pl.col("likelihood") >= cfg.pcutoff)
        .then(pl.col("y"))
        .otherwise(None)
        .alias("y"),
    )

    # Interpolate within each (individual, bodypart) group, forward/backward fill edges
    return keypoints_df.with_columns(
        pl.col("x")
        .interpolate()
        .forward_fill()
        .backward_fill()
        .over(["individual", "bodypart"]),
        pl.col("y")
        .interpolate()
        .forward_fill()
        .backward_fill()
        .over(["individual", "bodypart"]),
    )
