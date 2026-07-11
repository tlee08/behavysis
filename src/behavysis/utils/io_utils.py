"""IO Utils."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def read_files_parallel[T](
    files: list[Path], read_func: Callable[[Path], T]
) -> list[T]:
    """Read files in parallel using a thread pool."""
    with ThreadPoolExecutor() as executor:
        return list(executor.map(read_func, files))
