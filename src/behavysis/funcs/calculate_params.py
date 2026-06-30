"""Functions have the following format."""

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from behavysis.constants import LIKELIHOOD, SINGLE
from behavysis.models import ExperimentConfig
from behavysis.schemas import (
    KEYPOINTS_SCHEMA,
    check_bpts_exist,
    get_indivs_bpts,
    read_df,
)


class CalculateParamsFunc(Protocol):
    """Protocol for calculate_params functions."""

    def __call__(
        self,
        keypoints_fp: Path,
        config_fp: Path,
    ) -> None:
        """Protocol for calculate_params functions."""
        ...


def start_frame_from_likelihood(
    keypoints_fp: Path,
    config_fp: Path,
) -> None:
    """Determines start frame based on when subject likely entered the frame."""
    start_frame, _stop_frame = _calc_exists_from_likelihood(keypoints_fp, config_fp)
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config.auto.start_frame = start_frame
    config_fp.write_text(config.model_dump_json(indent=2))


def start_frame_from_csv(keypoints_fp: Path, config_fp: Path) -> None:
    """Determines start frame from timestamps in csv."""
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config_filt = config.user.calculate_params.start_frame_from_csv
    fps = config.auto.formatted_vid.fps
    csv_fp = config.get_ref(config_filt.csv_fp)
    name = config.get_ref(config_filt.name)
    assert fps != -1, (
        "fps not yet set. Please calculate fps first with `proj.get_vid_metadata`."
    )
    if name is None:
        name = keypoints_fp.stem
    start_times_df = pd.read_csv(csv_fp, index_col=0)
    start_times_df.index = start_times_df.index.astype(str)
    assert name in start_times_df.index.to_numpy(), (
        f"{name} not in {csv_fp}.\n"
        "Update `name` parameter in config file or check the start_frames csv file."
    )
    start_sec = start_times_df.loc[name][0]
    start_frame = int(np.round(start_sec * fps, 0))
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config.auto.start_frame = start_frame
    config_fp.write_text(config.model_dump_json(indent=2))


def stop_frame_from_likelihood(keypoints_fp: Path, config_fp: Path) -> None:
    """Determines stop frame based on when subject likely exited the frame."""
    _start_frame, stop_frame = _calc_exists_from_likelihood(keypoints_fp, config_fp)
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config.auto.stop_frame = stop_frame
    config_fp.write_text(config.model_dump_json(indent=2))


def stop_frame_from_dur(
    keypoints_fp: Path,  # noqa: ARG001
    config_fp: Path,
) -> None:
    """Calculates the end time from start_frame + experiment_duration."""
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config_filt = config.user.calculate_params.stop_frame_from_dur
    dur_sec = config.get_ref(config_filt.dur_sec)
    start_frame = config.auto.start_frame
    fps = config.auto.formatted_vid.fps
    total_frames = config.auto.formatted_vid.total_frames
    assert start_frame != -1, "start_frame is None. Please calculate start_frame first."
    assert fps != -1, (
        "fps not yet set. Please calculate fps first with `proj.get_vid_metadata`."
    )
    dur_frames = int(dur_sec * fps)
    stop_frame = start_frame + dur_frames
    if total_frames is None:
        logger.warning("The length of the video itself has not been calculated yet.")
    elif stop_frame > total_frames:
        logger.warning(
            "The user specified dur_sec in the config file is greater "
            "than the actual length of the video.",
        )
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config.auto.stop_frame = stop_frame
    config_fp.write_text(config.model_dump_json(indent=2))


def dur_frames_from_likelihood(keypoints_fp: Path, config_fp: Path) -> None:
    """Determines duration in frames from subject first to last seen."""
    start_frame, stop_frame = _calc_exists_from_likelihood(keypoints_fp, config_fp)
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config.auto.dur_frames = stop_frame - start_frame
    config_fp.write_text(config.model_dump_json(indent=2))


def px_per_mm(keypoints_fp: Path, config_fp: Path) -> None:
    """Calculates the pixels per mm conversion using calibration points."""
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config_filt = config.user.calculate_params.px_per_mm
    pt_a = config.get_ref(config_filt.pt_a)
    pt_b = config.get_ref(config_filt.pt_b)
    pcutoff = config.get_ref(config_filt.pcutoff)
    dist_mm = config.get_ref(config_filt.dist_mm)

    keypoints_df = read_df(keypoints_fp, KEYPOINTS_SCHEMA).fill_null(0)
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

    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config.auto.px_per_mm = px_per_mm_val
    config_fp.write_text(config.model_dump_json(indent=2))


def _calc_exists_from_likelihood(
    keypoints_fp: Path,
    config_fp: Path,
) -> tuple[int, int]:
    """Determine start/stop frames from likelihood thresholds."""
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config_filt = config.user.calculate_params.from_likelihood
    bpts = config.get_ref(config_filt.bodyparts)
    window_sec = config.get_ref(config_filt.window_sec)
    pcutoff = config.get_ref(config_filt.pcutoff)
    fps = config.auto.formatted_vid.fps
    assert fps != -1, (
        "fps not yet set. Please calculate fps first with `proj.get_vid_metadata`."
    )
    window_frames = int(np.round(fps * window_sec, 0))

    keypoints_df = read_df(keypoints_fp, KEYPOINTS_SCHEMA)
    check_bpts_exist(keypoints_df, bpts)
    indivs, _ = get_indivs_bpts(keypoints_df)

    # For each individual, compute median likelihood per frame across bodyparts
    all_exists = None
    for indiv in indivs:
        indiv_lhood = (
            keypoints_df.filter(
                pl.col("individual") == indiv,
                pl.col("bodypart").is_in(bpts),
            )
            .group_by("frame")
            .agg(pl.col("likelihood").median().alias("likelihood"))
            .sort("frame")
        )
        # Extract series for rolling window
        lhood_vals = indiv_lhood.select("likelihood").to_series().to_numpy()
        # Rolling mean
        rolling = (
            pd.Series(lhood_vals).rolling(window_frames, center=True).mean().to_numpy()
        )
        exists = rolling > pcutoff
        all_exists = exists.copy() if all_exists is None else all_exists & exists

    assert np.any(all_exists), "The subject was not detected in any frames."
    true_indices = np.flatnonzero(all_exists)
    start_frame = true_indices[0]
    stop_frame = true_indices[-1]
    return int(start_frame), int(stop_frame)
