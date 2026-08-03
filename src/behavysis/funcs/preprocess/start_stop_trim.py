"""Functions have the following format."""

import polars as pl

from behavysis.models import ExperimentConfig, ExperimentMetadata

# ═══════════════════════════════════════════════════════════════════════════════
# Functions
# ═══════════════════════════════════════════════════════════════════════════════


def start_stop_trim(
    keypoints_df: pl.DataFrame,
    config: ExperimentConfig,  # noqa: ARG001
    metadata: ExperimentMetadata,
) -> pl.DataFrame:
    """Filters the rows of a DLC formatted dataframe."""
    start_frame = metadata.require_start_frame()
    stop_frame = metadata.require_stop_frame()
    return keypoints_df.filter(
        pl.col("frame").is_between(start_frame, stop_frame),
    )
