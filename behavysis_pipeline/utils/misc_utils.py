import inspect
from enum import EnumType
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from behavysis_pipeline.utils.logging_utils import init_logger

logger = init_logger(__name__)


def make_colours(vals, cmap: str) -> np.ndarray:
    # If vals is an empty list, return colours_ls as an empty list
    if len(vals) == 0:
        return np.array([])
    # Encoding colours as 0, 1, 2, ... for each unique value
    colours_idx, _ = pd.factorize(vals)
    # Normalising to 0-1 (if only 1 unique value, it will be 0 div so setting values to 0)
    colours_idx = np.nan_to_num(colours_idx / colours_idx.max())
    # Getting corresponding colour for each item in `vals` list and from cmap
    colours_ls = plt.cm.get_cmap(cmap)(colours_idx)
    # Reassigning the order of the colours to be RGBA (not BGRA)
    colours_ls = colours_ls[:, [2, 1, 0, 3]]
    # Converting to (0, 255) range
    colours_ls = colours_ls * 255
    # Returning
    return colours_ls


def import_extra_error_func(extra_dep_name: str):
    def error_func(*args, **kwargs):
        raise ImportError(
            f"{extra_dep_name} dependency not installed.\n"
            f'Install with `pip install "microscopy_proc[{extra_dep_name}]"`'
        )

    return error_func


def enum2list(my_enum: EnumType) -> list[Any]:
    return [e.value for e in my_enum]


def enum2tuple(my_enum: EnumType) -> tuple[Any]:
    """
    Useful helper function to convert an Enum to a list of its values.
    Used in `check_df` and `init_df` functions.
    """
    return tuple(i.value for i in my_enum)


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


def get_current_function_name() -> str:
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
    if c_frame.f_back is None:
        return ""
    # Returning the name of the function that called this function
    return c_frame.f_back.f_code.co_name
