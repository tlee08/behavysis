"""Functions have the following format."""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel

from behavysis.models import ExperimentConfig, ExperimentMetadata

# ═══════════════════════════════════════════════════════════════════════════════
# Config Models
# ═══════════════════════════════════════════════════════════════════════════════


class StartFrameFromCsvConfig(BaseModel):
    """StartFrameFromCsvConfig."""

    csv_fp: Path


# ═══════════════════════════════════════════════════════════════════════════════
# Functions
# ═══════════════════════════════════════════════════════════════════════════════


def start_frame_from_csv(
    keypoints_df: pl.DataFrame,  # noqa: ARG001
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
) -> ExperimentMetadata:
    """Determines start frame from timestamps in csv."""
    # Read files
    cfg = config.require_calculate_parameters().require(
        "start_frame_from_csv",
        StartFrameFromCsvConfig,
    )
    # Read csv with start times
    start_times_df = pd.read_csv(cfg.csv_fp, index_col=0)
    start_times_df.index = start_times_df.index.astype(str)
    assert metadata.require_name() in start_times_df.index.to_numpy(), (
        f"{metadata.require_name()} not in {cfg.csv_fp}.\n"
        "Update `name` parameter in config file or check the start_frames csv file."
    )
    # Get start frame
    start_sec = start_times_df.loc[metadata.require_name()][0]
    start_frame = int(np.round(start_sec * metadata.require_fps(), 0))
    # Set start frame in metadata and save
    metadata.start_frame = start_frame
    return metadata
