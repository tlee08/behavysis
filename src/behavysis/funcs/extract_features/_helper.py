"""Helper funcs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import polars as pl

    from behavysis.models import ExperimentConfig, ExperimentMetadata


class ExtractFeaturesFunc(Protocol):
    """Protocol for extract features functions."""

    __name__: str

    def __call__(
        self,
        keypoints_df: pl.DataFrame,
        config: ExperimentConfig,
        metadata: ExperimentMetadata,
    ) -> pl.DataFrame:
        """Protocol for extract features functions."""
