"""Helper funcs."""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

if TYPE_CHECKING:
    from behavysis.models import ExperimentConfig, ExperimentMetadata


class AnalyseFunc(Protocol):
    """Protocol for analyse functions."""

    __name__: str

    def __call__(
        self,
        config: ExperimentConfig,
        metadata: ExperimentMetadata,
        **kwargs: object,
    ) -> list[AnalysisResult]:
        """Protocol for analyse functions."""


class AnalysisResult(BaseModel):
    """A serializable analysis result with self-contained save logic.

    Attributes:
    ----------
    relative_path : Path
        Path relative to the analysis output directory.
    result : object
        The computed data (DataFrame, numpy array, matplotlib figure, etc.).
    save_func : Callable[[Path, object], None]
        Function that saves ``result`` to the given absolute file path.
    """

    relative_path: Path
    result: object
    save_func: Callable[[Path, object], None]

    def save(self, dst_dir: Path) -> None:
        """Save result to ``dst_dir / self.relative_path``."""
        full_path = dst_dir / self.relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_func(full_path, self.result)
