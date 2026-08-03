"""Helper funcs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import polars as pl

if TYPE_CHECKING:
    import numpy as np

    from behavysis.models import AnalysisResult, ExperimentConfig, ExperimentMetadata


class AnalyseFunc(Protocol):
    """Protocol for analyse functions."""

    __name__: str

    def __call__(
        self,
        keypoints_df: pl.DataFrame,
        vid_frame: np.ndarray[tuple[int, int, int], np.dtype[np.uint8]],
        config: ExperimentConfig,
        metadata: ExperimentMetadata,
    ) -> list[AnalysisResult]:
        """Protocol for analyse functions."""


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _bodypart_avg_xy(
    df: pl.DataFrame,
    indiv: str,
    bpts: list[str],
) -> pl.DataFrame:
    """Average x and y coordinates across bodyparts per frame for an individual."""
    return (
        df.filter(
            pl.col("individual") == indiv,
            pl.col("bodypart").is_in(bpts),
        )
        .group_by("frame")
        .agg([pl.col("x").mean().alias("x"), pl.col("y").mean().alias("y")])
        .sort("frame")
    )
