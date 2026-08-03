"""Functions have the following format."""

from typing import Protocol

import polars as pl

from behavysis.models import ExperimentConfig, ExperimentMetadata


class CalculateParametersFunc(Protocol):
    """Protocol for calculate_parameters functions."""

    def __call__(
        self,
        keypoints_df: pl.DataFrame,
        config: ExperimentConfig,
        metadata: ExperimentMetadata,
    ) -> ExperimentMetadata:
        """Protocol for calculate_parameters functions."""
        ...
