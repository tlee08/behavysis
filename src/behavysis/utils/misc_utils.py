"""Miscellaneous utility functions."""

import contextlib
import gc
from collections.abc import Callable
from functools import wraps
from pathlib import Path

import torch
from loguru import logger


def pass_exception(
    _func: Callable,
    exception: type[BaseException] = Exception,
) -> Callable:
    """Don't raise exception."""

    def decorator(func: Callable) -> Callable:

        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            with contextlib.suppress(exception):
                return func(*args, **kwargs)

        return wrapper

    # Allow both @pass_exception and @pass_exception(exception=...)
    return decorator(_func) if _func else decorator


def check_files_exist(*fp_ls: Path) -> bool:
    """Quick-code check that given list of files must exist.

    If not, then logs, and returns False.
    """
    not_exists = [fp for fp in fp_ls if not fp.exists()]
    if len(not_exists) > 0:
        logger.warning("Files do not exist: {}", not_exists)
        return False
    return True


def get_gpu_device() -> str:
    """Use GPU when available, else CPU (keeps the Mac working)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_gpu_device_ids() -> list[int]:
    """Return GPU device IDs."""
    return list(range(torch.cuda.device_count()))


def clean_memory(_func: Callable) -> Callable:
    """Clear the GPU cache and garbage collector."""

    @wraps(_func)
    def wrapper(*args: object, **kwargs: object) -> object:
        # Run func
        res = _func(*args, **kwargs)
        # Clean
        torch.cuda.empty_cache()
        gc.collect()
        # Return
        return res

    return wrapper
