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

from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel

from behavysis.constants import BPTS_SIMBA
from behavysis.models import ExperimentConfig, ExperimentMetadata
from behavysis.schemas import check_bpts_exist

# ═══════════════════════════════════════════════════════════════════════════════
# Config Models
# ═══════════════════════════════════════════════════════════════════════════════


class RefineIdsConfig(BaseModel):
    """RefineIdsConfig."""

    marked: str = "marked"
    unmarked: str = "unmarked"
    marking: str = "marking"
    bodyparts: list[str] = BPTS_SIMBA
    window_sec: float = 0.5
    metric: Literal["current", "rolling", "binned"] = "current"


# ═══════════════════════════════════════════════════════════════════════════════
# Functions
# ═══════════════════════════════════════════════════════════════════════════════


def refine_ids(
    keypoints_df: pl.DataFrame,
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
) -> pl.DataFrame:
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
    cfg = config.require_preprocess().require("refine_ids", RefineIdsConfig)
    marked = cfg.marked
    unmarked = cfg.unmarked
    marking = cfg.marking
    window_sec = cfg.window_sec
    bpts = cfg.bodyparts
    metric = cfg.metric
    fps = metadata.require_fps()
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
    return _switch_identities(
        keypoints_df,
        switch_df.select(metric).to_series(),
        marked,
        unmarked,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Funcs
# ═══════════════════════════════════════════════════════════════════════════════


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
