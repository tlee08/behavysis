"""Functions have the following format."""

from pathlib import Path
from typing import Protocol

from behavysis.models import ExperimentConfig, ExperimentMetadata


class CalculateParamsFunc(Protocol):
    """Protocol for calculate_params functions."""

    def __call__(
        self,
        keypoints_fp: Path,
        config: ExperimentConfig,
        metadata: ExperimentMetadata,
    ) -> ExperimentMetadata:
        """Protocol for calculate_params functions."""
        ...
