"""Model adapters for sklearn and PyTorch classifiers.

SklearnAdapter wraps any sklearn-compatible estimator class with MinMaxScaler
+ feature selection + optional GridSearchCV. Hyperparameters are resolved from
the TrainingRecipe at fit time — the registry stores estimator classes, not
pre-configured instances.

TorchAdapter wraps a TorchModel with MinMaxScaler.

Serialisation:
- sklearn: model.joblib (joblib dump of entire adapter)
- torch:   model.pt (state_dict) + scaler.joblib (MinMaxScaler)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import joblib
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from behavysis.constants import ACTUAL, EXPERIMENT, FRAME, Array1D, Array2D

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl
    from sklearn.base import BaseEstimator

    from .config import TrainingRecipe
    from .torch.base import TorchModel

_META_COLS = [EXPERIMENT, FRAME, ACTUAL]


def select_features(
    x: Array2D,
    y: Array1D,
    config: TrainingRecipe,
) -> np.ndarray:
    """Return column indices to keep, fit on training data only.

    Drops low-variance columns, then optionally caps to the top
    ``max_features`` by random-forest importance. Returns all columns when
    ``feature_selection`` is disabled.
    """
    keep = np.arange(x.shape[1])
    if not config.feature_selection:
        return keep

    vt = VarianceThreshold(threshold=config.variance_threshold)
    vt.fit(x)
    keep = keep[vt.get_support()]

    if config.max_features is not None and len(keep) > config.max_features:
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(x[:, keep], y)
        top = np.argsort(rf.feature_importances_)[::-1][: config.max_features]
        keep = np.sort(keep[top])

    return keep


class BaseAdapter(ABC):
    """Abstract adapter with fit / predict."""

    framework: ClassVar[str]
    feature_mask: Array1D | None

    @abstractmethod
    def fit(
        self,
        df: pl.DataFrame,
        train_mask: np.ndarray,
        config: TrainingRecipe,
    ) -> pd.DataFrame:
        """Train on rows where ``train_mask`` is True.

        Returns per-epoch history (empty for sklearn).
        """
        ...

    @abstractmethod
    def predict(
        self,
        x: Array2D,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Return predicted probabilities for the given feature array."""
        ...

    @abstractmethod
    def save(self, version_dir: Path) -> None:
        """Persist model artifacts inside version_dir."""
        ...


class SklearnAdapter(BaseAdapter):
    """Adapter for sklearn estimators (RF, LogisticRegression, XGBoost, etc.).

    Takes an estimator class at construction. All hyperparameters are resolved
    from ``TrainingRecipe.hyperparameters`` at ``fit`` time via
    ``GridSearchCV``.

    Serialises as a single model.joblib blob (estimator + scaler).
    """

    framework: ClassVar[str] = "sklearn"

    def __init__(self, estimator_cls: type) -> None:
        self.estimator_cls = estimator_cls
        self.estimator: BaseEstimator | None = None
        self.scaler = MinMaxScaler()
        self.feature_mask: Array1D | None = None

    @property
    def resolved_hyperparameters(self) -> dict[str, object] | None:
        """Return ``best_params_`` from ``GridSearchCV``, or None if not fitted."""
        if self.estimator is not None and hasattr(self.estimator, "best_params_"):
            return self.estimator.best_params_
        return None

    def fit(
        self,
        df: pl.DataFrame,
        train_mask: np.ndarray,
        config: TrainingRecipe,
    ) -> pd.DataFrame:
        x = df.filter(train_mask).drop(_META_COLS).to_numpy()
        y = df.filter(train_mask)[ACTUAL].to_numpy()
        x = self.scaler.fit_transform(x)
        self.feature_mask = select_features(x, y, config)
        x = x[:, self.feature_mask]

        base = self.estimator_cls()
        gs = GridSearchCV(base, config.hyperparameters, scoring="f1", cv=3, n_jobs=-1)
        gs.fit(x, y)
        self.estimator = gs

        return pd.DataFrame(columns=pd.Index(["loss", "vloss"]))

    def predict(
        self,
        x: np.ndarray,
        batch_size: int = 256,
    ) -> np.ndarray:
        _ = batch_size
        x = self.scaler.transform(x)
        x = x[:, self.feature_mask]
        return self.estimator.predict_proba(x)[:, 1]

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
        self.feature_mask: np.ndarray | None = None

    def fit(
        self,
        df: pl.DataFrame,
        train_mask: np.ndarray,
        config: TrainingRecipe,
    ) -> pd.DataFrame:
        x = df.filter(train_mask).drop(_META_COLS).to_numpy()
        y = df.filter(train_mask)[ACTUAL].to_numpy()

        self.scaler = MinMaxScaler()
        x = self.scaler.fit_transform(x)
        self.feature_mask = select_features(x, y, config)
        nfeatures = len(self.feature_mask)

        self.model = self.model_factory(nfeatures)
        return self.model.fit(
            [x[:, self.feature_mask]],
            [y],
            [np.arange(x.shape[0])],
            batch_size=config.batch_size,
            epochs=config.epochs,
            val_split=config.val_split,
        )

    def predict(
        self,
        x: Array2D,
        batch_size: int = 256,
    ) -> np.ndarray:
        if self.scaler is None or self.model is None or self.feature_mask is None:
            msg = "Model not fitted. Call fit() or load_state() first."
            raise RuntimeError(msg)
        x = self.scaler.transform(x)[:, self.feature_mask]
        return self.model.predict(x, None, batch_size=batch_size)

    def save(self, version_dir: Path) -> None:
        if self.model is None or self.scaler is None or self.feature_mask is None:
            msg = "Cannot save unfitted torch model."
            raise RuntimeError(msg)
        version_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), version_dir / "model.pt")
        joblib.dump(self.scaler, version_dir / "scaler.joblib")
        np.save(version_dir / "feature_mask.npy", self.feature_mask)
        logger.info("Saved torch model to {}", version_dir)

    def load_state(self, version_dir: Path) -> None:
        """Reconstruct model + scaler + mask from version_dir artifacts.

        Requires self.model_factory to be set (from MODEL_REGISTRY
        instantiation before calling this method).
        """
        self.scaler = joblib.load(version_dir / "scaler.joblib")
        self.feature_mask = np.load(version_dir / "feature_mask.npy")
        nfeatures = len(self.feature_mask)
        self.model = self.model_factory(nfeatures)
        self.model.load_state_dict(
            torch.load(
                version_dir / "model.pt",
                map_location=torch.device("cpu"),
                weights_only=True,
            )
        )
        logger.info("Loaded torch model from {}", version_dir)
