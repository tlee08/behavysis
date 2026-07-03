"""Standalone preprocessing: column selection + MinMax scaling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler


def select_features(x: np.ndarray, start_col: int) -> np.ndarray:
    """Select derived features, dropping raw x-y-likelihood columns."""
    return x[:, start_col:]


class Scaler:
    """Wraps MinMaxScaler with save/load via joblib."""

    def __init__(self) -> None:
        self._scaler = MinMaxScaler()

    def fit(self, x: np.ndarray) -> Scaler:
        self._scaler.fit(x)
        return self

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self._scaler.fit_transform(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self._scaler.transform(x)

    def save(self, fp: Path) -> None:
        fp.parent.mkdir(parents=True, exist_ok=True)
        dump(self._scaler, fp)

    @classmethod
    def load(cls, fp: Path) -> Scaler:
        scaler = cls()
        scaler._scaler = load(fp)
        return scaler
