"""Helper funcs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import polars as pl
from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path

    from behavysis.models import ExperimentConfig, ExperimentMetadata


class AnalyseFunc(Protocol):
    """Protocol for analyse functions."""

    __name__: str

    def __call__(
        self,
        keypoints_fp: Path,
        formatted_vid_fp: Path,
        config: ExperimentConfig,
        metadata: ExperimentMetadata,
        dst_dir: Path,
    ) -> None:
        """Protocol for analyse functions."""


class AnalysisResult(BaseModel):
    """Analysis result of path, object, and saver func."""

    # TODO: swap out alaysis funcs return to this so it's testable.

    relative_path: Path
    obj: object
    save_func: Callable


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
