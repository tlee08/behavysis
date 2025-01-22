import inspect
from enum import EnumType
from typing import Any, Iterable

from behavysis_pipeline.utils.logging_utils import init_logger

logger = init_logger(__name__)


def import_extra_error_func(extra_dep_name: str):
    def error_func(*args, **kwargs):
        raise ImportError(
            f"{extra_dep_name} dependency not installed.\n"
            f'Install with `pip install "microscopy_proc[{extra_dep_name}]"`'
        )

    return error_func


def enum2tuple(my_enum: EnumType) -> tuple[Any]:
    return tuple(i.value for i in my_enum)  # type: ignore


def const2iter(x: Any, n: int) -> Iterable[Any]:
    """
    Iterates the object, `x`, `n` times.
    """
    for _ in range(n):
        yield x


def const2list(x: Any, n: int) -> list[Any]:
    """
    Iterates the list, `ls`, `n` times.
    """
    return [x for _ in range(n)]


def dictlists2listdicts(my_dict):
    """
    Converts a dict of lists to a list of dicts.
    """
    # Asserting that all values (lists) have same size
    n = len(list(my_dict.values())[0])
    for i in my_dict.values():
        assert len(i) == n
    # Making list of dicts
    return [{k: v[i] for k, v in my_dict.items()} for i in range(n)]


def listdicts2dictlists(my_list):
    """
    Converts a list of dicts to a dict of lists.
    """
    # Asserting that each dict has the same keys
    keys = my_list[0].keys()
    for i in my_list:
        assert i.keys() == keys
    # Making dict of lists
    return {k: [v[k] for v in my_list] for k in keys}


def get_current_funct_name() -> str:
    """
    Returns the name of the function that called this function.
    This is useful for debugging and dynamically changing function behavior
    (e.g. getting attributes according to the functions name).

    Note
    ----
    If this function is called from the main script (i.e. no function),
    it will return an empty string.
    """
    # Getting the current frame
    c_frame = inspect.currentframe()
    # If this function is called from the main script, return empty string
    if c_frame is None:
        return ""
    if c_frame.f_back is None:
        return ""
    return c_frame.f_back.f_code.co_name
