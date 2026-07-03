"""Helper funcs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import polars as pl
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np

    from behavysis.models import ExperimentConfig, ExperimentMetadata


class AnalyseFunc(Protocol):
    """Protocol for analyse functions."""

    __name__: str

    def __call__(
        self,
        keypoints_df: pl.DataFrame,
        vid_frame: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
        config: ExperimentConfig,
        metadata: ExperimentMetadata,
    ) -> list[AnalysisResult]:
        """Protocol for analyse functions."""


class AnalysisResult(BaseModel):
    """Analysis result of path, object, and saver func."""

    relative_path: Path
    result: object
    save_func: Callable[[Path, object], None]

    def save(self, dst_dir: Path) -> None:
        """Save."""
        self.save_func(dst_dir / self.relative_path, self.result)


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
