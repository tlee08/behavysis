"""Miscellaneous utility functions."""

import contextlib
from collections.abc import Callable
from functools import wraps

import torch


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


def get_gpu_device() -> str:
    """Use GPU when available, else CPU (keeps the Mac working)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_gpu_device_ids() -> list[int]:
    """Return GPU device IDs."""
    return list(range(torch.cuda.device_count()))
