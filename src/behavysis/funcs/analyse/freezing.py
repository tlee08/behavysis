"""Analysis functions operating on Polars long-form keypoints DataFrames."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from pydantic import BaseModel, PositiveFloat

from behavysis.constants import DF_IO_FORMAT, FBF
from behavysis.models import AnalysisResult
from behavysis.schemas import ANALYSIS_SCHEMA, write_df
from behavysis.transforms.analysis import summary_binned_behaviour
from behavysis.transforms.behaviour import vect2bouts
from behavysis.transforms.keypoint import check_bpts_exist, get_indivs_bpts

if TYPE_CHECKING:
    from behavysis.models import ExperimentConfig, ExperimentMetadata


class FreezingConfig(BaseModel):
    """FreezingConfig."""

    bodyparts: list[str]
    window_sec: PositiveFloat = 2.0
    thresh_mm: PositiveFloat = 5.0
    smoothing_sec: PositiveFloat = 0.2


def freezing(
    keypoints_df: pl.DataFrame,
    vid_frame: np.ndarray,  # noqa: ARG001
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
) -> list[AnalysisResult]:
    """Determines frames where the subject is frozen (movement below threshold)."""
    name = metadata.require_name()

    cfg = config.require_analyse().require("freezing", FreezingConfig)
    bpts = cfg.bodyparts
    thresh_mm = cfg.thresh_mm
    smoothing_sec = cfg.smoothing_sec
    window_sec = cfg.window_sec

    thresh_px = thresh_mm / metadata.require_px_per_mm()
    smoothing_frames = int(smoothing_sec * metadata.require_fps())
    window_frames = int(np.round(metadata.require_fps() * window_sec))

    check_bpts_exist(keypoints_df, bpts)
    indivs, _ = get_indivs_bpts(keypoints_df)

    all_dfs = []

    for indiv in indivs:
        indiv_df = keypoints_df.filter(pl.col("individual") == indiv)
        frames = indiv_df.select("frame").unique().sort("frame").to_series()

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

        n = len(frames)
        is_freezing = np.ones(n, dtype=bool)
        for deltas in deltas_list:
            is_freezing &= np.array(deltas[:n]) < thresh_px

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

    return [
        AnalysisResult(
            relative_path=Path(FBF) / f"{name}.{DF_IO_FORMAT}",
            result=analysis_df,
            save_func=lambda fp, obj: write_df(obj, fp, ANALYSIS_SCHEMA),
        ),
        *summary_binned_behaviour(
            analysis_df,
            name,
            metadata.require_fps(),
            config.require_analyse().bins_sec_ls,
            config.require_analyse().custom_bins_sec_ls,
        ),
    ]
