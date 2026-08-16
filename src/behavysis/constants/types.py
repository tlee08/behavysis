"""Typing aliases."""

import numpy as np

type Array1D = np.ndarray[tuple[int], np.dtype[np.float64]]
type Array2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]

type Array1DInt = np.ndarray[tuple[int], np.dtype[np.int64]]
