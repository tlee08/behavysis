"""Functions have the following format.

Parameters
----------
dlc_fp : str
    The input video filepath.
dst_fp : str
    The output video filepath.
config_fp : str
    The JSON config filepath.
overwrite : bool
    Whether to overwrite the output file (if it exists).

Returns:
-------
str
    Description of the function's outcome.

"""

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
    start_frame = metadata.start_frame
    stop_frame = metadata.stop_frame
    return keypoints_df.filter(
        pl.col("frame").is_between(start_frame, stop_frame),
    )
