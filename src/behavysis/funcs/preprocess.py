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

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from behavysis.constants import SINGLE
from behavysis.models import ExperimentConfig
from behavysis.schemas import KEYPOINTS_SCHEMA, check_bpts_exist, read_df, write_df
from behavysis.utils.io_utils import file_exists_msg


class PreprocessFunc(Protocol):
    """Protocol for preprocess functions."""

    def __call__(
        self,
        src_fp: Path,
        dst_fp: Path,
        config_fp: Path,
        *,
        overwrite: bool,
    ) -> None:
        """Protocol for preprocess functions."""


def start_stop_trim(
    src_fp: Path,
    dst_fp: Path,
    config_fp: Path,
    *,
    overwrite: bool,
) -> None:
    """Filters the rows of a DLC formatted dataframe.

    Includes only rows within the start
    and end time of the experiment, given a corresponding config dict.

    Parameters
    ----------
    dlc_fp : str
        The file path of the input DLC formatted dataframe.
    dst_fp : Path
        The file path of the output trimmed dataframe.
    config_fp : Path
        The file path of the config dict.
    overwrite : bool
        If True, overwrite the output file if it already exists. If False, skip
        if the output file already exists.

    Returns:
    -------
    str
        An outcome message indicating the result of the trimming process.

    Notes:
    -----
    The config file must contain the following parameters:
    ```
    - user
        - preprocess
            - start_stop_trim
                - start_frame: int
                - stop_frame: int
    ```
    """
    if not overwrite and dst_fp.exists():
        logger.warning(file_exists_msg(dst_fp))
        return
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    start_frame = config.auto.start_frame
    stop_frame = config.auto.stop_frame
    keypoints_df = read_df(src_fp, KEYPOINTS_SCHEMA)
    keypoints_df = keypoints_df.filter(
        pl.col("frame").is_between(start_frame, stop_frame),
    )
    write_df(keypoints_df, dst_fp, KEYPOINTS_SCHEMA)


def interpolate_stationary(
    src_fp: Path,
    dst_fp: Path,
    config_fp: Path,
    *,
    overwrite: bool,
) -> None:
    """If the point detection (above a certain threshold) is below a certain proportion.

    Then the x and y coordinates are set to the given values (usually corners).
    Otherwise, does nothing (encouraged to run Preprocess.interpolate afterwards).

    """
    if not overwrite and dst_fp.exists():
        logger.warning(file_exists_msg(dst_fp))
        return
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config_filt_ls = config.user.preprocess.interpolate_stationary
    width_px = config.auto.formatted_vid.width_px
    height_px = config.auto.formatted_vid.height_px
    if width_px <= 0 or height_px <= 0:
        msg = (
            f"Video dimensions not set for experiment.\n"
            f"  width_px={width_px}, height_px={height_px}\n"
            f"  Run proj.format_video() first to set these values."
        )
        raise ValueError(msg)

    keypoints_df = read_df(src_fp, KEYPOINTS_SCHEMA)

    for config_filt in config_filt_ls:
        bodypart = config_filt.bodypart
        pcutoff = config_filt.pcutoff
        pcutoff_all = config_filt.pcutoff_all
        x_px = config_filt.x * width_px
        y_px = config_filt.y * height_px

        # Get likelihood for single individual + bodypart
        mask = (pl.col("individual") == SINGLE) & (pl.col("bodypart") == bodypart)
        is_detected = (
            keypoints_df.filter(mask)
            .select(
                pl.col("likelihood") >= pcutoff,
            )
            .to_series()
        )

        if is_detected.mean() < pcutoff_all:
            # Set x, y, likelihood for all frames matching this bodypart
            keypoints_df = keypoints_df.with_columns(
                pl.when(mask).then(pl.lit(x_px)).otherwise(pl.col("x")).alias("x"),
                pl.when(mask).then(pl.lit(y_px)).otherwise(pl.col("y")).alias("y"),
                pl.when(mask)
                .then(pl.lit(pcutoff))
                .otherwise(pl.col("likelihood"))
                .alias("likelihood"),
            )
            logger.info(
                f"{bodypart} is detected in less than {pcutoff_all} of the video."
                f" Setting x and y coordinates to ({x_px}, {y_px}).",
            )
        else:
            logger.info(
                f"{bodypart} is detected in more than {pcutoff_all} of the video."
                " No need for stationary interpolation.",
            )

    write_df(keypoints_df, dst_fp, KEYPOINTS_SCHEMA)


def interpolate(
    src_fp: Path,
    dst_fp: Path,
    config_fp: Path,
    *,
    overwrite: bool,
) -> None:
    """Smooths noticeable jitter of points.

    Where the likelihood (and accuracy) of
    a point's coordinates are low
    (e.g., when the subject's head goes out of view).
    It does this by linearly interpolating the frames
    of a body part that are below a given likelihood pcutoff.

    Notes:
    -----
    The config file must contain the following parameters:
    ```
    - user
        - preprocess
            - interpolate
                - pcutoff: float
    ```
    """
    if not overwrite and dst_fp.exists():
        logger.warning(file_exists_msg(dst_fp))
        return
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config_filt = config.user.preprocess.interpolate

    keypoints_df = read_df(src_fp, KEYPOINTS_SCHEMA)

    # Fill any null likelihoods with 0
    keypoints_df = keypoints_df.with_columns(
        pl.col("likelihood").fill_null(0),
    )

    # Set x and y to null where likelihood is below pcutoff
    keypoints_df = keypoints_df.with_columns(
        pl.when(pl.col("likelihood") >= config_filt.pcutoff)
        .then(pl.col("x"))
        .otherwise(None)
        .alias("x"),
        pl.when(pl.col("likelihood") >= config_filt.pcutoff)
        .then(pl.col("y"))
        .otherwise(None)
        .alias("y"),
    )

    # Interpolate within each (individual, bodypart) group, forward/backward fill edges
    keypoints_df = keypoints_df.with_columns(
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

    write_df(keypoints_df, dst_fp, KEYPOINTS_SCHEMA)


def refine_ids(src_fp: Path, dst_fp: Path, config_fp: Path, *, overwrite: bool) -> None:
    """Ensures that the identity is correctly tracked for maDLC.

    Assumes interpolate_points has already been run.

    Notes:
    -----
    The config file must contain the following parameters:
    ```
    - user
        - preprocess
            - refine_ids
                - marked: str
                - unmarked: str
                - marking: str
                - window_sec: float
                - metric: ["current", "rolling", "binned"]
    ```
    """
    if not overwrite and dst_fp.exists():
        logger.warning(file_exists_msg(dst_fp))
        return

    keypoints_df = read_df(src_fp, KEYPOINTS_SCHEMA)

    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config_filt = config.user.preprocess.refine_ids
    marked = config.get_ref(config_filt.marked)
    unmarked = config.get_ref(config_filt.unmarked)
    marking = config.get_ref(config_filt.marking)
    window_sec = config.get_ref(config_filt.window_sec)
    bpts = config.get_ref(config_filt.bodyparts)
    metric = config.get_ref(config_filt.metric)
    fps = config.auto.formatted_vid.fps
    window_frames = int(np.round(fps * window_sec, 0))

    # Validate individuals and marking exist
    available_indivs = keypoints_df.select("individual").unique().to_series().to_list()
    available_bpts = keypoints_df.select("bodypart").unique().to_series().to_list()
    for column, level, available in [
        (marked, "individuals", available_indivs),
        (unmarked, "individuals", available_indivs),
        (marking, "bodyparts", available_bpts),
    ]:
        if column not in available:
            msg = (
                f'The value in the config file, "{column}"'
                f" is not present in the {level} of the DLC file."
            )
            raise ValueError(msg)

    check_bpts_exist(keypoints_df, bpts)

    # Calculate distances between each individual and the marking
    mark_dists_df = _get_mark_dists_df(keypoints_df, marked, unmarked, [marking], bpts)

    # Get switch decisions
    switch_df = _get_id_switch_df(mark_dists_df, window_frames, marked, unmarked)

    # Apply identity switches
    switched = _switch_identities(
        keypoints_df,
        switch_df.select(metric).to_series(),
        marked,
        unmarked,
    )

    write_df(switched, dst_fp, KEYPOINTS_SCHEMA)


def _get_mark_dists_df(
    keypoints_df: pl.DataFrame,
    marked_indiv: str,
    unmarked_indiv: str,
    mark_pts: list[str],
    bpts: list[str],
) -> pl.DataFrame:
    """Calculate distances between individuals and marking for identity refinement.

    Parameters
    ----------
    keypoints_df : pl.DataFrame
        Keypoints dataframe in long form (KEYPOINTS_SCHEMA).
    marked_indiv : str
        Name of marked individual.
    unmarked_indiv : str
        Name of unmarked individual.
    mark_pts : list[str]
        Marking bodypoints.
    bpts : list[str]
        Bodypoints to average for distance calculation.

    Returns:
    -------
    pl.DataFrame
        DataFrame with frame, mark_x, mark_y, marked_x,marked_y,
        unmarked_x, unmarked_y.
    """
    # Average marking bodypart coordinates per frame
    mark_xy = (
        keypoints_df.filter(
            pl.col("individual") == "single",
            pl.col("bodypart").is_in(mark_pts),
        )
        .group_by("frame")
        .agg([pl.col("x").mean().alias("mark_x"), pl.col("y").mean().alias("mark_y")])
        .sort("frame")
    )

    # Average marked individual bodypart coordinates per frame
    marked_xy = (
        keypoints_df.filter(
            pl.col("individual") == marked_indiv,
            pl.col("bodypart").is_in(bpts),
        )
        .group_by("frame")
        .agg(
            [
                pl.col("x").mean().alias("marked_x"),
                pl.col("y").mean().alias("marked_y"),
            ],
        )
        .sort("frame")
    )

    # Average unmarked individual bodypart coordinates per frame
    unmarked_xy = (
        keypoints_df.filter(
            pl.col("individual") == unmarked_indiv,
            pl.col("bodypart").is_in(bpts),
        )
        .group_by("frame")
        .agg(
            [
                pl.col("x").mean().alias("unmarked_x"),
                pl.col("y").mean().alias("unmarked_y"),
            ],
        )
        .sort("frame")
    )

    # Join all together on frame
    dists = mark_xy.join(marked_xy, on="frame").join(unmarked_xy, on="frame")

    # Calculate Euclidean distances
    return dists.with_columns(
        (
            (pl.col("marked_x") - pl.col("mark_x")).pow(2)
            + (pl.col("marked_y") - pl.col("mark_y")).pow(2)
        )
        .sqrt()
        .alias("marked_dist"),
        (
            (pl.col("unmarked_x") - pl.col("mark_x")).pow(2)
            + (pl.col("unmarked_y") - pl.col("mark_y")).pow(2)
        )
        .sqrt()
        .alias("unmarked_dist"),
    )


def _get_id_switch_df(
    mark_dists_df: pl.DataFrame,
    window_frames: int,
    marked: str,
    unmarked: str,
) -> pl.DataFrame:
    """Calculate identity switch decisions using current, rolling, and binned metrics.

    Parameters
    ----------
    mark_dists_df : pl.DataFrame
        DataFrame with ``marked_dist`` and ``unmarked_dist`` columns.
    window_frames : int
        Window size in frames for rolling/binned calculations.
    marked : str
        Name of marked individual (unused, kept for API compatibility).
    unmarked : str
        Name of unmarked individual (unused, kept for API compatibility).

    Returns:
    -------
    pl.DataFrame
        DataFrame with ``frame``, ``current``, ``rolling``, ``binned`` columns.
    """
    _ = marked, unmarked  # explicitly unused in new API

    switch_df = mark_dists_df.select(
        "frame",
        (pl.col("marked_dist") > pl.col("unmarked_dist")).alias("current"),
    ).sort("frame")

    # Rolling mode over window
    # Convert to pandas for rolling mode (Polars has no rolling mode)
    current_pd = switch_df.select("current").to_series().to_pandas()

    def _rolling_mode(x: "np.ndarray") -> bool:
        vals, counts = np.unique(x, return_counts=True)
        return vals[counts.argmax()]

    rolling_pd = (
        current_pd.rolling(window_frames, min_periods=1)
        .apply(
            _rolling_mode,
            raw=True,
        )
        .astype(bool)
    )

    # Binned mode
    frames = switch_df.select("frame").to_series().to_numpy()
    bins = np.arange(frames.min(), frames.max() + window_frames, window_frames)
    binned_labels = pd.cut(frames, bins=bins, labels=bins[1:], include_lowest=True)
    current_pd.index = binned_labels
    binned_pd = (
        current_pd.groupby(level=0)
        .transform(
            lambda x: x.mode().iloc[0] if len(x) > 0 else False,
        )
        .astype(bool)
    )

    # Reconstruct as Polars DataFrame
    return pl.DataFrame(
        {
            "frame": switch_df.select("frame").to_series(),
            "current": switch_df.select("current").to_series(),
            "rolling": pl.Series(rolling_pd.values),
            "binned": pl.Series(binned_pd.values),
        },
    )


def _switch_identities(
    keypoints_df: pl.DataFrame,
    is_switch: pl.Series | np.ndarray,
    marked_indiv: str,
    unmarked_indiv: str,
) -> pl.DataFrame:
    """Swap individual identities where is_switch is True.

    In Polars long form, this means flipping the ``individual`` column values
    between ``marked_indiv`` and ``unmarked_indiv`` on frames where the switch
    is active.
    """
    if isinstance(is_switch, np.ndarray):
        is_switch = pl.Series("is_switch", is_switch)

    # Build a frame→switch lookup
    frames = keypoints_df.select("frame").unique().sort("frame").to_series()
    switch_lookup = pl.DataFrame({"frame": frames, "is_switch": is_switch})

    return (
        keypoints_df.join(switch_lookup, on="frame")
        .with_columns(
            pl.when(
                pl.col("is_switch") & (pl.col("individual") == marked_indiv),
            )
            .then(pl.lit(unmarked_indiv))
            .when(
                pl.col("is_switch") & (pl.col("individual") == unmarked_indiv),
            )
            .then(pl.lit(marked_indiv))
            .otherwise(pl.col("individual"))
            .alias("individual"),
        )
        .drop("is_switch")
    )
