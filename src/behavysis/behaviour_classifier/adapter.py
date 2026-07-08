"""Model adapters for sklearn and PyTorch classifiers.

SklearnAdapter wraps any sklearn estimator with MinMaxScaler.
TorchAdapter wraps a TorchModel with MinMaxScaler.

Serialisation:
- sklearn: model.joblib (joblib dump of entire adapter)
- torch:   model.pt (state_dict) + scaler.joblib (MinMaxScaler)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import joblib
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

if TYPE_CHECKING:
    from .config import TrainingRecipe
    from .torch.base import TorchModel


class BaseAdapter(ABC):
    """Abstract adapter with fit / predict."""

    framework: ClassVar[str]

    @abstractmethod
    def fit(
        self,
        x_ls: list[np.ndarray],
        y_ls: list[np.ndarray],
        train_idx: list[np.ndarray],
        config: TrainingRecipe,
    ) -> pd.DataFrame:
        """Train on train_idx subsets. Returns per-epoch history (empty for sklearn)."""
        ...

    @abstractmethod
    def predict(
        self,
        x: np.ndarray,
        index: np.ndarray | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Return predicted probabilities for indexed rows."""
        ...

    @abstractmethod
    def save(self, version_dir: Path) -> None:
        """Persist model artifacts inside version_dir."""
        ...


class SklearnAdapter(BaseAdapter):
    """Adapter for sklearn estimators (RF, LogisticRegression, etc.).

    Serialises as a single model.joblib blob (estimator + scaler).
    """

    framework: ClassVar[str] = "sklearn"

    def __init__(self, estimator: object) -> None:
        self.estimator = estimator
        self.scaler = MinMaxScaler()

    def fit(
        self,
        x_ls: list[np.ndarray],
        y_ls: list[np.ndarray],
        train_idx: list[np.ndarray],
        _config: TrainingRecipe,
    ) -> pd.DataFrame:
        x_train = [x[idx] for x, idx in zip(x_ls, train_idx, strict=True)]
        y_train = [y[idx] for y, idx in zip(y_ls, train_idx, strict=True)]
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

    def save(self, version_dir: Path) -> None:
        version_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, version_dir / "model.joblib")
        logger.info("Saved sklearn model to {}", version_dir)


class TorchAdapter(BaseAdapter):
    """Adapter for PyTorch models with lazy architecture instantiation.

    model_factory: Callable[[int], TorchModel] — takes nfeatures,
    returns a fresh architecture. Model created at fit() or load_state()
    time when nfeatures is known.

    Serialises as model.pt (state_dict) + scaler.joblib.
    """

    framework: ClassVar[str] = "torch"

    def __init__(self, model_factory) -> None:
        self.model_factory = model_factory
        self.model: TorchModel | None = None
        self.scaler: MinMaxScaler | None = None

    def fit(
        self,
        x_ls: list[np.ndarray],
        y_ls: list[np.ndarray],
        train_idx: list[np.ndarray],
        config: TrainingRecipe,
    ) -> pd.DataFrame:
        x_train = [x[idx] for x, idx in zip(x_ls, train_idx, strict=True)]
        y_train = [y[idx] for y, idx in zip(y_ls, train_idx, strict=True)]
        X = np.concatenate(x_train, axis=0)
        y = np.concatenate(y_train, axis=0)

        self.scaler = MinMaxScaler()
        X = self.scaler.fit_transform(X)
        nfeatures = X.shape[1]

        self.model = self.model_factory(nfeatures)
        return self.model.fit(
            [X],
            [y],
            [np.arange(X.shape[0])],
            batch_size=config.batch_size,
            epochs=config.epochs,
            val_split=config.val_split,
        )

    def predict(
        self,
        x: np.ndarray,
        index: np.ndarray | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        if self.scaler is None or self.model is None:
            msg = "Model not fitted. Call fit() or load_state() first."
            raise RuntimeError(msg)
        X = self.scaler.transform(x)
        idx = index if index is not None else np.arange(X.shape[0])
        return self.model.predict(X, idx, batch_size=batch_size)

    def save(self, version_dir: Path) -> None:
        if self.model is None or self.scaler is None:
            msg = "Cannot save unfitted torch model."
            raise RuntimeError(msg)
        version_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), version_dir / "model.pt")
        joblib.dump(self.scaler, version_dir / "scaler.joblib")
        logger.info("Saved torch model to {}", version_dir)

    def load_state(self, version_dir: Path) -> None:
        """Reconstruct model + scaler from version_dir artifacts.

        Requires self.model_factory to be set (from MODEL_REGISTRY
        instantiation before calling this method).
        """
        self.scaler = joblib.load(version_dir / "scaler.joblib")
        nfeatures = self.scaler.n_features_in_
        self.model = self.model_factory(nfeatures)
        self.model.load_state_dict(
            torch.load(
                version_dir / "model.pt",
                map_location=torch.device("cpu"),
                weights_only=True,
            )
        )
        logger.info("Loaded torch model from {}", version_dir)
