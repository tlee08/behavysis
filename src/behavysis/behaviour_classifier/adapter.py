"""Model adapters for sklearn and PyTorch classifiers.

SklearnAdapter receives a ``pipeline_builder(config) -> ImbPipeline``
from the MODEL_REGISTRY.  At fit time it wraps the pipeline in
``GridSearchCV``; at predict time it delegates to the fitted pipeline.
The pipeline definition itself lives entirely in the registry.

TorchAdapter wraps a TorchModel with MinMaxScaler + standalone feature
selection.

Serialisation:
- sklearn: model.joblib (joblib dump of pipeline, strip GridSearchCV wrapper)
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
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from behavysis.constants import ACTUAL, EXPERIMENT, FRAME, Array2D

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl
    from imblearn.pipeline import Pipeline as ImbPipeline

    from .config import TrainingRecipe
    from .torch.base import TorchModel

_META_COLS = [EXPERIMENT, FRAME, ACTUAL]


class BaseAdapter(ABC):
    """Abstract adapter with fit / predict."""

    framework: ClassVar[str]

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

    @abstractmethod
    def predict(self, x: Array2D) -> np.ndarray:
        """Return predicted probabilities for the given feature array."""

    @abstractmethod
    def save(self, version_dir: Path) -> None:
        """Persist model artifacts inside version_dir."""


class SklearnAdapter(BaseAdapter):
    """Thin wrapper around an imblearn Pipeline + GridSearchCV.

    The pipeline is built by calling ``self.pipeline_builder(config)``
    at fit time — the builder is defined in the MODEL_REGISTRY, keeping
    all pipeline logic in one place.  GridSearchCV is then fitted and
    the adapter delegates predict / feature access to the pipeline.
    """

    framework: ClassVar[str] = "sklearn"

    def __init__(
        self,
        pipeline: ImbPipeline,
    ) -> None:
        self.pipeline = pipeline

    @property
    def pipe(self) -> ImbPipeline:
        """The fitted Pipeline, unwrapped from GridSearchCV if needed."""
        return getattr(self.pipeline, "best_estimator_", self.pipeline)

    @property
    def resolved_hyperparameters(self) -> dict[str, object] | None:
        """Return ``best_params_`` from ``GridSearchCV``, or None if not fitted."""
        if hasattr(self.pipeline, "best_params_"):
            return self.pipeline.best_params_
        return None

    def fit(
        self, df: pl.DataFrame, train_mask: np.ndarray, config: TrainingRecipe
    ) -> pd.DataFrame:
        x = df.filter(train_mask).drop(_META_COLS).to_numpy()
        y = df.filter(train_mask)[ACTUAL].to_numpy()

        gs = GridSearchCV(
            self.pipeline,
            config.hyperparameters,
            scoring="f1",
            cv=3,
            n_jobs=1,
            verbose=1,
        ).fit(x, y)

        self.pipeline = gs

        return pd.DataFrame(columns=pd.Index(["loss", "vloss"]))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(x)[:, 1]

    def save(self, version_dir: Path) -> None:
        version_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipe, version_dir / "model.joblib")
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
        self, df: pl.DataFrame, train_mask: np.ndarray, config: TrainingRecipe
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
    ) -> np.ndarray:
        if self.scaler is None or self.model is None or self.feature_mask is None:
            msg = "Model not fitted. Call fit() or load_state() first."
            raise RuntimeError(msg)
        x = self.scaler.transform(x)[:, self.feature_mask]
        return self.model.predict(x, None, batch_size=256)

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
