"""Functions have the following format."""

import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel, PositiveFloat

from behavysis.models import ExperimentConfig, ExperimentMetadata
from behavysis.transforms import check_bpts_exist, get_indivs_bpts

# ═══════════════════════════════════════════════════════════════════════════════
# Config Models
# ═══════════════════════════════════════════════════════════════════════════════


class FromLikelihoodConfig(BaseModel):
    """FromLikelihoodConfig."""

    bodyparts: list[str]
    window_sec: PositiveFloat
    pcutoff: PositiveFloat


# ═══════════════════════════════════════════════════════════════════════════════
# Functions
# ═══════════════════════════════════════════════════════════════════════════════


def start_frame_from_likelihood(
    keypoints_df: pl.DataFrame,
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
) -> ExperimentMetadata:
    """Determines start frame based on when subject likely entered the frame."""
    # Calculate start-stop frames from likelihood
    _start_frame, _stop_frame = _calc_exists_from_likelihood(
        keypoints_df,
        config,
        metadata,
    )
    # Set start frame in metadata and save
    metadata.start_frame = _start_frame
    return metadata


def stop_frame_from_likelihood(
    keypoints_df: pl.DataFrame,
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
) -> ExperimentMetadata:
    """Determines stop frame based on when subject likely exited the frame."""
    # Calculate start-stop frames from likelihood
    _start_frame, _stop_frame = _calc_exists_from_likelihood(
        keypoints_df,
        config,
        metadata,
    )
    # Set stop frame in metadata and save
    metadata.stop_frame = _stop_frame
    return metadata


def dur_frames_from_likelihood(
    keypoints_df: pl.DataFrame,
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
) -> ExperimentMetadata:
    """Determines duration in frames from subject first to last seen."""
    # Calculate start-stop frames from likelihood
    _start_frame, _stop_frame = _calc_exists_from_likelihood(
        keypoints_df,
        config,
        metadata,
    )
    # Set stop frame in metadata and save
    metadata.dur_frames = _stop_frame - _start_frame
    return metadata


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Funcs
# ═══════════════════════════════════════════════════════════════════════════════


def _calc_exists_from_likelihood(
    keypoints_df: pl.DataFrame,
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
) -> tuple[int, int]:
    """Determine start/stop frames from likelihood thresholds."""
    cfg = config.require_calculate_parameters().require(
        "from_likelihood",
        FromLikelihoodConfig,
    )
    window_frames = int(np.round(metadata.require_fps() * cfg.window_sec, 0))

    check_bpts_exist(keypoints_df, cfg.bodyparts)
    indivs, _ = get_indivs_bpts(keypoints_df)

    # For each individual, compute median likelihood per frame across bodyparts
    all_exists = None
    for indiv in indivs:
        indiv_lhood = (
            keypoints_df.filter(
                pl.col("individual") == indiv,
                pl.col("bodypart").is_in(cfg.bodyparts),
            )
            .group_by("frame")
            .agg(pl.col("likelihood").median().alias("likelihood"))
            .sort("frame")
        )
        # Extract series for rolling window
        lhood_vals = indiv_lhood.select("likelihood").to_series().to_numpy()
        # Rolling mean
        exists = (
            pd.Series(lhood_vals).rolling(window_frames, center=True).mean().to_numpy()
        ) > cfg.pcutoff

        all_exists = exists if all_exists is None else all_exists & exists

    if all_exists is None:
        msg = "No individuals."
        raise TypeError(msg)
    if not np.any(all_exists):
        msg = "The subject was not detected in any frames."
        raise ValueError(msg)
    true_indices = np.flatnonzero(all_exists)
    start_frame = true_indices[0]
    stop_frame = true_indices[-1]
    return int(start_frame), int(stop_frame)
