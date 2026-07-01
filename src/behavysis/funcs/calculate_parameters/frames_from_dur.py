"""Functions have the following format."""

from pathlib import Path

from loguru import logger
from pydantic import BaseModel, PositiveFloat

from behavysis.models import ExperimentConfig, ExperimentMetadata

# ═══════════════════════════════════════════════════════════════════════════════
# Config Models
# ═══════════════════════════════════════════════════════════════════════════════


class StopFrameFromDurConfig(BaseModel):
    """StopFrameFromDurConfig."""

    dur_sec: PositiveFloat


# ═══════════════════════════════════════════════════════════════════════════════
# Functions
# ═══════════════════════════════════════════════════════════════════════════════


def stop_frame_from_dur(
    keypoints_fp: Path,  # noqa: ARG001
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
) -> ExperimentMetadata:
    """Calculates the end time from start_frame + experiment_duration."""
    # Read files
    cfg = config.require_calculate_parameters().require(
        "stop_frame_from_dur",
        StopFrameFromDurConfig,
    )
    # Calculate stop frame from start frame + duration
    dur_frames = int(cfg.dur_sec * metadata.require_fps())
    stop_frame = metadata.require_start_frame() + dur_frames
    if stop_frame > metadata.require_total_frames():
        logger.warning(
            "The user specified dur_sec in the config file is greater "
            "than the actual length of the video.",
        )
    # Set stop frame in metadata and save
    metadata.stop_frame = stop_frame
    return metadata
