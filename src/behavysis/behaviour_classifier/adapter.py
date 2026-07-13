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
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from behavysis.constants import (
    ACTUAL,
    BEHAVIOUR,
    BOUT_ID,
    EXPERIMENT,
    FRAME,
    PRED,
    PROB,
    Array1D,
    Array2D,
)
from behavysis.schemas import BEHAVIOUR_PREDICTED_SCHEMA

from .config import TrainingRecipe
from .data import label_bouts, stratified_split_by_group
from .torch._helper import select_features

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Self

    from sklearn.base import BaseEstimator
    from sklearn.model_selection._search import BaseSearchCV

    from .torch.base import TorchModel

_META_COLS = [EXPERIMENT, FRAME, ACTUAL, BOUT_ID]


class BaseAdapter(ABC):
    """Abstract adapter with fit / predict."""

    framework: ClassVar[str]
    config_fp: Path

    def _read_config(self) -> TrainingRecipe:
        """Read config."""
        return TrainingRecipe.read_yaml(self.config_fp)

    def _write_config(self, config: TrainingRecipe) -> None:
        """Read config."""
        return config.write_yaml(self.config_fp)

    def _features(self, df: pl.DataFrame) -> Array2D:
        return df.drop(_META_COLS, strict=False).to_numpy().astype(np.float32)

    def _labels(self, df: pl.DataFrame) -> Array1D:
        return df[ACTUAL].to_numpy()

    @abstractmethod
    def fit(self, df: pl.DataFrame) -> pd.DataFrame:
        """Train on rows where ``train_mask`` is True.

        Returns per-epoch history (empty for sklearn).
        """

    @abstractmethod
    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return predicted probabilities for the given feature array."""

    @abstractmethod
    def save(self, model_dir: Path) -> None:
        """Persist model artifacts inside dst_dir."""

    @classmethod
    @abstractmethod
    def load(cls, model_dir: Path) -> Self:
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

    def __init__(self, search: BaseSearchCV, config_fp: Path) -> None:
        """Init."""
        self.search = search
        self.config_fp = config_fp
        self.model: BaseEstimator | None = None

    def fit(self, df: pl.DataFrame) -> pd.DataFrame:
        """Fit."""
        # 0. Read config
        config = self._read_config()
        # Hyperparameter selection stage
        # 1. Compute bout_id once, on the full ordered frame (correct boundaries).
        df = label_bouts(df)
        # 2. Row-level, prevalence-preserving downsample for the search.
        sub_df = df
        if len(df) > config.downsample_n:
            idx = np.arange(len(df))
            sub_idx, _ = train_test_split(
                idx,
                train_size=config.downsample_n,
                stratify=self._labels(df),
                random_state=config.seed,
            )
            sub_df = df[sub_idx]
        # 3. Grouped-by-bout_id CV on surviving rows → no CV leakage.
        self.search.refit = False
        self.search.fit(
            self._features(sub_df),
            self._labels(sub_df),
            groups=sub_df[BOUT_ID].to_numpy(),
        )
        # Full training stage
        # 4. Make train-val split by bouts (to programatically find best pcutoff)
        train_idx, val_idx = stratified_split_by_group(
            df, config.val_split, BOUT_ID, config.seed
        )
        train_df = df[train_idx]
        val_df = df[val_idx]
        # 5. Refit best pipeline on the train_df
        self.model = clone(self.search.estimator).set_params(**self.search.best_params_)
        self.model.fit(self._features(train_df), self._labels(train_df))
        # 6. Find best pcutoff with val_df and update config with best pcutoff value
        y_df = self.predict(val_df).with_columns(val_df[ACTUAL], val_df[BOUT_ID])
        _, recall, thresholds = precision_recall_curve(y_df[ACTUAL], y_df[PROB])
        config.pcutoff = (
            thresholds[(recall[:-1] >= config.target_recall)][-1]
            if np.any(recall[:-1] >= config.target_recall)
            else 0.001
        )
        config.pcutoff = 0.01
        self._write_config(config)
        # Return
        return pd.DataFrame(columns=pd.Index(["loss", "vloss"]))

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Predict."""
        if self.model is None:
            msg = "model not yet trained."
            raise ValueError(msg)
        # Read config
        config = self._read_config()
        # Prepare
        frame = df.get_column(FRAME)
        # Predict
        prob = self.model.predict_proba(self._features(df))[:, 1]
        # Construct df and return
        return pl.DataFrame(
            {
                FRAME: frame,
                BEHAVIOUR: pl.lit(config.behaviour_name),
                PROB: prob,
                PRED: prob > config.pcutoff,
            },
            schema=BEHAVIOUR_PREDICTED_SCHEMA,
        )

    def save(self, model_dir: Path) -> None:
        """Save."""
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.search, model_dir / "search.joblib")
        logger.info("Saved sklearn model to {}", model_dir)

    @classmethod
    def load(cls, model_dir: Path) -> Self:
        """Load."""
        search = joblib.load(model_dir / "search.joblib")
        config_fp = model_dir / "config.yaml"
        inst = cls(search, config_fp)
        inst.model = clone(inst.search.estimator).set_params(**inst.search.best_params_)
        return inst

    def cv_summary(self) -> dict | None:
        """Return the search best score (computed on the subsample)."""
        best_score = getattr(self.search, "best_score_", None)
        if best_score is None:
            return None
        return {"cv_average_precision_subsampled": float(best_score)}


class TorchAdapter(BaseAdapter):
    """Adapter for PyTorch models with lazy architecture instantiation."""

    framework: ClassVar[str] = "torch"

    def __init__(self, model: TorchModel, config_fp: Path) -> None:
        """Init."""
        self.model: TorchModel = model
        self.config_fp = config_fp
        self.scaler: MinMaxScaler = MinMaxScaler()
        self.feature_mask: np.ndarray = np.ndarray([])

    def fit(self, df: pl.DataFrame) -> pd.DataFrame:
        """Fit."""
        config = self._read_config()
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
        """Predict."""
        config = self._read_config()
        # Prepare
        x = df.drop(_META_COLS, strict=False).to_numpy()
        frame = df.get_column(FRAME)
        # Predict
        x = self.scaler.transform(x)[:, self.feature_mask]
        prob = self.model.predict(x, None, batch_size=256)
        # Construct df and return
        return pl.DataFrame(
            {
                FRAME: frame,
                BEHAVIOUR: pl.lit(config.behaviour_name),
                PROB: prob,
                PRED: prob > config.pcutoff,
            },
            schema=BEHAVIOUR_PREDICTED_SCHEMA,
        )

    def save(self, model_dir: Path) -> None:
        """Save."""
        if self.model is None or self.scaler is None or self.feature_mask is None:
            msg = "Cannot save unfitted torch model."
            raise RuntimeError(msg)
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_dir / "model.pt")
        joblib.dump(self.scaler, model_dir / "scaler.joblib")
        np.save(model_dir / "feature_mask.npy", self.feature_mask)
        logger.info("Saved torch model to {}", model_dir)

    @classmethod
    def load(cls, model_dir: Path) -> Self:
        """Reconstruct model + scaler + mask from version_dir artifacts.

        Requires self.model_factory to be set (from MODEL_REGISTRY
        instantiation before calling this method).
        """
        model = model.load_state_dict(
            torch.load(
                model_dir / "model.pt",
                map_location=torch.device("cpu"),
                weights_only=True,
            )
        )
        inst = cls(model)
        inst.scaler = joblib.load(model_dir / "scaler.joblib")
        inst.feature_mask = np.load(model_dir / "feature_mask.npy")
        logger.info("Loaded torch model from {}", model_dir)
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
