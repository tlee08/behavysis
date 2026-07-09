"""AnalysisResult model for pipeline analyse functions."""

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel


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
