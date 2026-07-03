"""Miscellaneous utility functions."""

import contextlib
from collections.abc import Callable
from functools import wraps


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
