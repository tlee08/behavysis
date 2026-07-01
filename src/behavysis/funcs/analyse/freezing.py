"""Analysis functions operating on Polars long-form keypoints DataFrames.

Functions have the following format:
    func(keypoints_fp, formatted_vid_fp, dst_dir, config_fp) -> None
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from pydantic import BaseModel, PositiveFloat

from behavysis.constants import BPTS_SIMBA, DF_IO_FORMAT, FBF
from behavysis.schemas import (
    ANALYSIS_SCHEMA,
    KEYPOINTS_SCHEMA,
    check_bpts_exist,
    get_indivs_bpts,
    read_df,
    summary_binned_behaviour,
    vect2bouts,
    write_df,
)

if TYPE_CHECKING:
    from pathlib import Path

    from behavysis.models import ExperimentConfig, ExperimentMetadata


# ═══════════════════════════════════════════════════════════════════════════════
# Config Models
# ═══════════════════════════════════════════════════════════════════════════════


class FreezingConfig(BaseModel):
    """FreezingConfig."""

    window_sec: PositiveFloat = 2.0
    thresh_mm: PositiveFloat = 5.0
    smoothing_sec: PositiveFloat = 0.2
    bodyparts: list[str] = BPTS_SIMBA


# ═══════════════════════════════════════════════════════════════════════════════
# Functions
# ═══════════════════════════════════════════════════════════════════════════════


def freezing(
    keypoints_fp: Path,
    formatted_vid_fp: Path,  # noqa: ARG001
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
    dst_dir: Path,
) -> None:
    """Determines frames where the subject is frozen (movement below threshold)."""
    name = keypoints_fp.stem

    cfg = config.require_analyse().require("freezing", FreezingConfig)
    bpts = cfg.bodyparts
    thresh_mm = cfg.thresh_mm
    smoothing_sec = cfg.smoothing_sec
    window_sec = cfg.window_sec

    thresh_px = thresh_mm / metadata.require_px_per_mm()
    smoothing_frames = int(smoothing_sec * metadata.require_fps())
    window_frames = int(np.round(metadata.require_fps() * window_sec))

    keypoints_df = read_df(keypoints_fp, KEYPOINTS_SCHEMA)
    check_bpts_exist(keypoints_df, bpts)
    indivs, _ = get_indivs_bpts(keypoints_df)

    all_dfs = []

    for indiv in indivs:
        indiv_df = keypoints_df.filter(pl.col("individual") == indiv)
        frames = indiv_df.select("frame").unique().sort("frame").to_series()

        # For each bodypart, compute per-frame delta
        deltas_list = []
        for bpt in bpts:
            bpt_df = indiv_df.filter(pl.col("bodypart") == bpt).sort("frame")
            delta_x = bpt_df.select("x").to_series().diff().fill_null(0)
            delta_y = bpt_df.select("y").to_series().diff().fill_null(0)
            delta = (delta_x.pow(2) + delta_y.pow(2)).sqrt()
            smoothed = delta.rolling_mean(
                window_size=smoothing_frames,
                min_samples=1,
                center=True,
            )
            deltas_list.append(smoothed.to_list())

        # Freezing if ALL bodyparts are below threshold
        n = len(frames)
        is_freezing = np.ones(n, dtype=bool)
        for deltas in deltas_list:
            is_freezing &= np.array(deltas[:n]) < thresh_px

        # Filter out short freezing bouts
        freezing_np = is_freezing.astype(np.int32)
        bouts = vect2bouts(pl.Series(freezing_np) == 1)
        for row in bouts.iter_rows(named=True):
            if row["dur"] < window_frames:
                freezing_np[row["start"] : row["stop"] + 1] = 0

        all_dfs.append(
            pl.DataFrame(
                {
                    "frame": frames,
                    "individual": indiv,
                    "measure": "freezing",
                    "value": freezing_np.astype(np.float64),
                },
                schema=ANALYSIS_SCHEMA,
            ),
        )

    analysis_df = pl.concat(all_dfs)

    fbf_fp = dst_dir / FBF / f"{name}.{DF_IO_FORMAT}"
    write_df(analysis_df, fbf_fp, ANALYSIS_SCHEMA)

    summary_binned_behaviour(
        analysis_df,
        dst_dir,
        name,
        metadata.require_fps(),
        config.require_analyse().bins_sec_ls,
        config.require_analyse().custom_bins_sec_ls,
    )
