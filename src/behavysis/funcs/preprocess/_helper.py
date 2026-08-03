"""Functions have the following format."""

from typing import Protocol

import polars as pl

from behavysis.models import ExperimentConfig, ExperimentMetadata


class PreprocessFunc(Protocol):
    """Protocol for preprocess functions."""

    def __call__(
        self,
        keypoints_df: pl.DataFrame,
        config: ExperimentConfig,
        metadata: ExperimentMetadata,
    ) -> pl.DataFrame:
        """Protocol for preprocess functions."""
