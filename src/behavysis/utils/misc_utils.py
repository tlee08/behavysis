from enum import EnumType
from typing import Any

import numpy as np


def enum2tuple(my_enum: EnumType) -> tuple[Any]:
    return tuple(i.value for i in my_enum)


def enum2list(my_enum: EnumType) -> list[Any]:
    return [i.value for i in my_enum]


def listofvects2array(*list_of_vects):
    """Converts a set of a list of vectors to a numpy array.

    ```
    *list_of_vects = [
        [
            (a, b, c),  # Vector 1
            (d, e, f),  # Vector 2
        ],
        [
            (x, y, z),  # Vector 1
            (i, j, k),  # Vector 2
        ]
    ]
    --> np.array([
        [1, a, x],
        [1, b, y],
        [1, c, z],
        [2, d, i],
        [2, e, j],
        [2, f, k],
    ])
    ```

    Helpful when each element in the list refers to a dataset.
    Returns a numpy array of (dataset_index, list_1_el, list_2_el).
    """
    # Assert that all vects across the set have the same length
    if len(list_of_vects) == 0:
        return np.zeros(shape=(0, 0))
    for list_of_vects_i in zip(*list_of_vects, strict=False):
        if len(list_of_vects_i) == 0:
            continue
        for v in list_of_vects_i[1:]:
            assert v.shape[0] == list_of_vects_i[0].shape[0]
    # Getting the lengths of each list of vects
    lengths_ls = [i.shape[0] for i in list_of_vects[0]]
    # Making the array
    return np.concatenate(
        [
            np.stack((np.repeat(i, vects[0]), *vects[1:]), axis=1)
            for i, vects in enumerate(zip(lengths_ls, *list_of_vects, strict=False))
        ],
        axis=0,
    )


def array2listofvect(arr, vect_index):
    """Inverse of listofvects2array, except chooses only one of the vects."""
    return [arr[arr[:, 0] == i, vect_index] for i in np.sort(np.unique(arr[:, 0]))]
