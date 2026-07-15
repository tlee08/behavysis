"""Functions have the following format."""

import polars as pl
from loguru import logger
from pydantic import BaseModel, PositiveFloat, PositiveInt

from behavysis.constants import SINGLE
from behavysis.models import ExperimentConfig, ExperimentMetadata

# ═══════════════════════════════════════════════════════════════════════════════
# Config Models
# ═══════════════════════════════════════════════════════════════════════════════


class InterpolateStationaryConfig(BaseModel):
    """InterpolateStationaryConfig."""

    bodypart: str = "bodypart"
    pcutoff: PositiveFloat = 0.8
    pcutoff_all: PositiveFloat = 0.6
    x: PositiveInt = 0
    y: PositiveInt = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Functions
# ═══════════════════════════════════════════════════════════════════════════════


def interpolate_stationary(
    keypoints_df: pl.DataFrame,
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
) -> pl.DataFrame:
    """If the point detection (above a certain threshold) is below a certain proportion.

    Then the x and y coordinates are set to the given values (usually corners).
    Otherwise, does nothing (encouraged to run Preprocess.interpolate afterwards).
    """
    cfg_ls = config.require_preprocess().require_list(
        "interpolate_stationary",
        InterpolateStationaryConfig,
    )
    width_px = metadata.require_width_px()
    height_px = metadata.require_height_px()

    for cfg in cfg_ls:
        x_px = cfg.x * width_px
        y_px = cfg.y * height_px

        # Get likelihood for single individual + bodypart
        mask = (pl.col("individual") == SINGLE) & (pl.col("bodypart") == cfg.bodypart)
        is_detected = (
            keypoints_df.filter(mask)
            .select(
                pl.col("likelihood") >= cfg.pcutoff,
            )
            .to_series()
        )

        if is_detected.mean() < cfg.pcutoff_all:
            # Set x, y, likelihood for all frames matching this bodypart
            keypoints_df = keypoints_df.with_columns(
                pl.when(mask).then(pl.lit(x_px)).otherwise(pl.col("x")).alias("x"),
                pl.when(mask).then(pl.lit(y_px)).otherwise(pl.col("y")).alias("y"),
                pl.when(mask)
                .then(pl.lit(cfg.pcutoff))
                .otherwise(pl.col("likelihood"))
                .alias("likelihood"),
            )
            logger.info(
                f"{cfg.bodypart} is detected in less than "
                f"{cfg.pcutoff_all} of the video."
                f" Setting x and y coordinates to ({x_px}, {y_px}).",
            )
        else:
            logger.info(
                f"{cfg.bodypart} is detected in more than "
                f"{cfg.pcutoff_all} of the video."
                " No need for stationary interpolation.",
            )

    return keypoints_df
