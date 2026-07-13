"""Typing aliases."""

from typing import Literal

import numpy as np

type Array1D = np.ndarray[tuple[int], np.dtype[np.float64]]
type Array2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]

type ModelStrOptions = Literal["sklearn", "torch"]
