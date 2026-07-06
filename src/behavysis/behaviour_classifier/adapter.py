"""Classifier adapters: SklearnAdapter and TorchAdapter."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler

from .torch.base import TorchModel


class BaseAdapter(ABC):
    """Self-contained classifier with model + scaler. Serialised via joblib."""

    @abstractmethod
    def fit(
        self,
        x_ls: list[np.ndarray],
        y_ls: list[np.ndarray],
        index_ls: list[np.ndarray],
        config: object,
    ) -> pd.DataFrame:
        """Train on indexed samples. Returns training history DataFrame."""
        ...

    @abstractmethod
    def predict(
        self,
        x: np.ndarray,
        index: np.ndarray | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Return predicted probabilities for indexed samples."""
        ...


class SklearnAdapter(BaseAdapter):
    """Wraps any sklearn estimator. Row-by-row (no temporal context)."""

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator
        self.scaler = MinMaxScaler()

    def fit(
        self,
        x_ls: list[np.ndarray],
        y_ls: list[np.ndarray],
        index_ls: list[np.ndarray],
        config: object,
    ) -> pd.DataFrame:
        x_train = [x[idx] for x, idx in zip(x_ls, index_ls, strict=True)]
        y_train = [y[idx] for y, idx in zip(y_ls, index_ls, strict=True)]
        X = np.concatenate(x_train, axis=0)
        y = np.concatenate(y_train, axis=0)
        X = self.scaler.fit_transform(X)
        self.estimator.fit(X, y)
        return pd.DataFrame(columns=["loss", "vloss"])

    def predict(
        self,
        x: np.ndarray,
        index: np.ndarray | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        _ = batch_size
        idx = index if index is not None else np.arange(x.shape[0])
        x_scaled = self.scaler.transform(x[idx])
        return self.estimator.predict_proba(x_scaled)[:, 1]

    def save(self, fp: Path) -> None:
        fp.parent.mkdir(parents=True, exist_ok=True)
        dump(self, fp)

    @classmethod
    def load(cls, fp: Path) -> BaseAdapter:
        return load(fp)


class TorchAdapter(BaseAdapter):
    """Wraps a TorchModel factory. Model built at fit time when nfeatures known."""

    def __init__(self, model_factory: Callable[[int], TorchModel]) -> None:
        self.model_factory = model_factory
        self.model: TorchModel | None = None
        self.scaler = MinMaxScaler()

    def fit(
        self,
        x_ls: list[np.ndarray],
        y_ls: list[np.ndarray],
        index_ls: list[np.ndarray],
        config: object,
    ) -> pd.DataFrame:
        batch_size = getattr(config, "batch_size", 256)
        epochs = getattr(config, "epochs", 100)
        val_split = getattr(config, "val_split", 0.2)

        x_train = [x[idx] for x, idx in zip(x_ls, index_ls, strict=True)]
        y_train = [y[idx] for y, idx in zip(y_ls, index_ls, strict=True)]

        X = np.concatenate(x_train, axis=0)
        self.scaler.fit(X)

        x_scaled = [self.scaler.transform(x) for x in x_train]
        nfeatures = x_scaled[0].shape[1]

        self.model = self.model_factory(nfeatures)
        idx_per_video = [np.arange(len(x)) for x in x_scaled]

        return self.model.fit(
            x_ls=x_scaled,
            y_ls=y_train,
            index_ls=idx_per_video,
            batch_size=batch_size,
            epochs=epochs,
            val_split=val_split,
        )

    def predict(
        self,
        x: np.ndarray,
        index: np.ndarray | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        x_scaled = self.scaler.transform(x)
        return self.model.predict(x_scaled, index=index, batch_size=batch_size)

    def save(self, fp: Path) -> None:
        fp.parent.mkdir(parents=True, exist_ok=True)
        dump(self, fp)

    @classmethod
    def load(cls, fp: Path) -> BaseAdapter:
        return load(fp)
