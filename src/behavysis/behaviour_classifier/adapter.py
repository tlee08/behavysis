"""Model adapters for sklearn and PyTorch classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Literal

import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

from behavysis.behaviour_classifier.data import label_bouts
from behavysis.behaviour_classifier.torch._helper import select_features
from behavysis.constants import ACTUAL, BOUT_ID, EXPERIMENT, FRAME, PROB
from behavysis.schemas import BEHAVIOUR_PROBABILITY_SCHEMA

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Self

    from sklearn.base import BaseEstimator

    from .config import TrainingRecipe
    from .torch.base import TorchModel

_META_COLS = [EXPERIMENT, FRAME, ACTUAL]


class BaseAdapter(ABC):
    """Abstract adapter with fit / predict."""

    framework: ClassVar[str]

    @abstractmethod
    def fit(self, df: pl.DataFrame, config: TrainingRecipe) -> pd.DataFrame:
        """Train on rows where ``train_mask`` is True.

        Returns per-epoch history (empty for sklearn).
        """

    @abstractmethod
    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return predicted probabilities for the given feature array."""

    @abstractmethod
    def save(self, dst_dir: Path) -> None:
        """Persist model artifacts inside dst_dir."""

    @classmethod
    @abstractmethod
    def load(cls, src_dir: Path) -> Self:
        """Load model artifacts."""

    def cv_summary(self) -> dict | None:
        """Return cross-validation summary, or None if unavailable."""
        return None


class SklearnAdapter(BaseAdapter):
    """Sklearn adapter.

    The pipeline is built by calling ``self._builder(config)``
    at fit time — the builder is defined in the MODEL_REGISTRY, keeping
    all pipeline logic in one place.  GridSearchCV is then fitted and
    the adapter delegates predict / feature access to the pipeline.
    """

    framework: ClassVar[str] = "sklearn"

    def __init__(self, model: BaseEstimator) -> None:
        self.model = model

    def fit(self, df: pl.DataFrame, config: TrainingRecipe) -> pd.DataFrame:
        # Prepare
        x = df.drop(_META_COLS, strict=False).to_numpy()
        y = df[ACTUAL].to_numpy()
        groups = label_bouts(df)[BOUT_ID].to_numpy()
        # Train
        self.model.fit(x, y, groups=groups)
        # Return
        return pd.DataFrame(columns=pd.Index(["loss", "vloss"]))

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        # Prepare
        x = df.drop(_META_COLS, strict=False).to_numpy()
        frame = df.get_column(FRAME)
        # Predict
        prob = self.model.predict_proba(x)[:, 1]
        # Return
        return pl.DataFrame(
            {FRAME: frame, PROB: prob}, schema=BEHAVIOUR_PROBABILITY_SCHEMA
        )

    def save(self, dst_dir: Path) -> None:
        dst_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, dst_dir / "model.joblib")
        logger.info("Saved sklearn model to {}", dst_dir)

    @classmethod
    def load(cls, src_dir: Path) -> Self:
        pipeline = joblib.load(src_dir / "model.joblib")
        return cls(pipeline)

    def cv_summary(self) -> dict | None:
        """Return the RandomizedSearchCV best score, if fitted."""
        best_score = getattr(self.model, "best_score_", None)
        if best_score is None:
            return None
        return {"cv_average_precision": float(best_score)}


class TorchAdapter(BaseAdapter):
    """Adapter for PyTorch models with lazy architecture instantiation."""

    framework: ClassVar[str] = "torch"

    def __init__(self, model: TorchModel) -> None:
        self.model: TorchModel = model
        self.scaler: MinMaxScaler = MinMaxScaler()
        self.feature_mask: np.ndarray = np.ndarray([])

    def fit(self, df: pl.DataFrame, config: TrainingRecipe) -> pd.DataFrame:
        # Prepare
        x = df.drop(_META_COLS, strict=False).to_numpy()
        y = df[ACTUAL].to_numpy()
        # Preprocess
        x = self.scaler.fit_transform(x)
        self.feature_mask = select_features(
            x, y, config.variance_threshold, config.max_features
        )
        nfeatures = len(self.feature_mask)
        # Train
        return self.model.fit(
            [x[:, self.feature_mask]],
            [y],
            [np.arange(x.shape[0])],
            batch_size=config.batch_size,
            epochs=config.epochs,
            val_split=config.val_split,
        )

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        # Prepare
        x = df.drop(_META_COLS, strict=False).to_numpy()
        frame = df.get_column(FRAME)
        # Predict
        x = self.scaler.transform(x)[:, self.feature_mask]
        prob = self.model.predict(x, None, batch_size=256)
        # Return
        return pl.DataFrame(
            {FRAME: frame, PROB: prob}, schema=BEHAVIOUR_PROBABILITY_SCHEMA
        )

    def save(self, dst_dir: Path) -> None:
        if self.model is None or self.scaler is None or self.feature_mask is None:
            msg = "Cannot save unfitted torch model."
            raise RuntimeError(msg)
        dst_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), dst_dir / "model.pt")
        joblib.dump(self.scaler, dst_dir / "scaler.joblib")
        np.save(dst_dir / "feature_mask.npy", self.feature_mask)
        logger.info("Saved torch model to {}", dst_dir)

    @classmethod
    def load(cls, src_dir: Path) -> Self:
        """Reconstruct model + scaler + mask from version_dir artifacts.

        Requires self.model_factory to be set (from MODEL_REGISTRY
        instantiation before calling this method).
        """
        model = model.load_state_dict(
            torch.load(
                src_dir / "model.pt",
                map_location=torch.device("cpu"),
                weights_only=True,
            )
        )
        inst = cls(model)
        inst.scaler = joblib.load(src_dir / "scaler.joblib")
        inst.feature_mask = np.load(src_dir / "feature_mask.npy")
        logger.info("Loaded torch model from {}", src_dir)
        return inst


# ── registry ─────────────────────────────────────────────────────────

type ModelStrOptions = Literal["sklearn", "torch"]

MODEL_TYPES_TO_STRING: dict[type[BaseAdapter], ModelStrOptions] = {
    SklearnAdapter: "sklearn",
    TorchAdapter: "torch",
}
MODEL_TYPES_TO_CLASS: dict[ModelStrOptions, type[BaseAdapter]] = {
    v: k for k, v in MODEL_TYPES_TO_STRING.items()
}
