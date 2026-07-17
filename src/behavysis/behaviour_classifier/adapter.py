"""Model adapters for sklearn and PyTorch classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tabpfn import TabPFNClassifier, load_fitted_tabpfn_model, save_fitted_tabpfn_model
from xgboost import XGBClassifier

from behavysis.constants import (
    ACTUAL,
    BEHAVIOUR,
    BOUT_ID,
    EXPERIMENT,
    FRAME,
    PRED,
    PROB,
    Array1D,
)
from behavysis.schemas import BEHAVIOUR_PREDICTED_SCHEMA
from behavysis.utils import get_gpu_device

from .config import TrainingRecipe
from .data import (
    agg_eval_df_by_bouts,
    df_get_features,
    df_get_labels,
    df_resample,
    label_bouts,
    smooth_preds,
    stratified_split_by_group,
)
from .torch._helper import select_features

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Self

    from sklearn.model_selection._search import BaseSearchCV

    from .torch.base import TorchModel


class BaseAdapter(ABC):
    """Abstract adapter with fit / predict."""

    framework: ClassVar[str]
    config_fp: Path

    def _read_config(self) -> TrainingRecipe:
        """Read config."""
        return TrainingRecipe.read_yaml(self.config_fp)

    def _write_config(self, config: TrainingRecipe) -> None:
        """Write config."""
        return config.write_yaml(self.config_fp)

    @abstractmethod
    def fit(self, df: pl.DataFrame) -> pd.DataFrame:
        """Train on rows where ``train_mask`` is True.

        Returns per-epoch history (empty for sklearn).
        """

    @abstractmethod
    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return predicted probabilities + binary preds with smoothing."""

    @abstractmethod
    def save(self) -> None:
        """Persist model artifacts inside dst_dir."""

    @classmethod
    @abstractmethod
    def load(cls, config_fp: Path) -> Self:
        """Load model artifacts."""


class SklearnAdapter(BaseAdapter):
    """Sklearn adapter."""

    framework: ClassVar[str] = "sklearn"

    def __init__(self, config_fp: Path, search: BaseSearchCV) -> None:
        """Init."""
        self.config_fp = config_fp
        self.search = search
        self.model: Pipeline | None = None

    def fit(self, df: pl.DataFrame) -> pd.DataFrame:
        """Fit."""
        # 1. Read config
        config = self._read_config()
        # Hyperparameter selection stage
        # 2. Sort the df by EXPERIMENT, FRAME. Then compute bout_id
        df = df.sort([EXPERIMENT, FRAME])
        df = label_bouts(df)
        # 4. hyperparameter selection on df. CV grouped-by-bout_id
        self.search.refit = False
        self.search.fit(
            df_get_features(df),
            df_get_labels(df),
            groups=df[BOUT_ID].to_numpy(),
        )
        # Full training stage
        # 5. Make train-val split by bouts (to programatically find best pcutoff)
        train_idx, val_idx = stratified_split_by_group(
            df, config.val_split, BOUT_ID, config.seed
        )
        train_df = df[train_idx]
        # 6. Refit best pipeline on train_df
        self.model = clone(self.search.estimator).set_params(**self.search.best_params_)
        self.model.fit(df_get_features(train_df), df_get_labels(train_df))
        # 7. Find best pcutoff with val_idx and update config with best pcutoff value
        # Use per-bouts eval instead of per-frames eval
        y_df = self.predict(df).with_columns(df[EXPERIMENT], df[ACTUAL], df[BOUT_ID])
        y_val_df = y_df[val_idx]
        y_val_bouts_df = agg_eval_df_by_bouts(y_val_df)
        _, recall, thresholds = precision_recall_curve(
            y_val_bouts_df[ACTUAL], y_val_bouts_df[PROB], drop_intermediate=True
        )
        config.pcutoff = float(
            thresholds[(recall[:-1] >= config.target_recall)][-1]
            if np.any(recall[:-1] >= config.target_recall)
            else 0.001
        )
        self._write_config(config)
        # Return
        return pd.DataFrame(columns=pd.Index(["loss", "vloss"]))

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return predicted probabilities + binary preds with smoothing."""
        # Check model exists
        if self.model is None:
            msg = "model not yet trained."
            raise ValueError(msg)
        # Get configs
        config = self._read_config()
        # Predict
        frame = df.get_column(FRAME)
        prob = self.model.predict_proba(df_get_features(df))[:, 1]
        # Construct df
        y_df = pl.DataFrame(
            {
                FRAME: frame,
                BEHAVIOUR: config.behaviour_name,
                PROB: prob,
                PRED: prob > config.pcutoff,
            },
            schema=BEHAVIOUR_PREDICTED_SCHEMA,
        )
        return smooth_preds(y_df, config.smoothing_frames, "median")

    def save(self) -> None:
        """Save."""
        # Check model exists
        if self.model is None:
            msg = "model not yet trained."
            raise ValueError(msg)
        # Save
        model_dir = self.config_fp.parent
        joblib.dump(self.search, model_dir / "search.joblib")
        joblib.dump(self.model, model_dir / "model.joblib")

    @classmethod
    def load(cls, config_fp: Path) -> Self:
        """Load."""
        model_dir = config_fp.parent
        inst = cls(model_dir / "config.yaml", joblib.load(model_dir / "search.joblib"))
        inst.model = joblib.load(model_dir / "model.joblib")
        return inst


class XgboostAdapter(SklearnAdapter):
    """XGBoost adapter.

    Different from SklearnAdapter to save/load portable serialisation
    across systems.
    """

    framework: ClassVar[str] = "xgboost"

    def save(self) -> None:
        """Save.

        Must save XGBoost model as a .ubj so it serialisable to all machines.
        """
        # Check model exists
        if self.model is None:
            msg = "model not yet trained."
            raise ValueError(msg)
        # clf is XGBoost, must first move to CPU before serialising
        preprocess: Pipeline = Pipeline(self.model.steps[:-1])
        clf: XGBClassifier = self.model.steps[-1][1]
        # Save
        model_dir = self.config_fp.parent
        joblib.dump(self.search, model_dir / "search.joblib")
        joblib.dump(preprocess, model_dir / "preprocess.joblib")
        clf.save_model(model_dir / "clf.ubj")

    @classmethod
    def load(cls, config_fp: Path) -> Self:
        """Load.

        Must load XGBoost model as a .ubj so it serialisable to all machines.
        """
        model_dir = config_fp.parent
        # Instatiate
        inst = cls(model_dir / "config.yaml", joblib.load(model_dir / "search.joblib"))
        # Load pipelin
        preprocess: Pipeline = joblib.load(model_dir / "preprocess.joblib")
        # Load model
        clf = XGBClassifier(device=get_gpu_device())
        clf.load_model(model_dir / "clf.ubj")
        # Reconstruct pipeline
        inst.model = Pipeline([*preprocess.steps, ("clf", clf)])
        # Return
        return inst


class TabPFNAdapter(BaseAdapter):
    """Adapter for TabPFN."""

    framework: ClassVar[str] = "tabpfn"

    def __init__(
        self,
        config_fp: Path,
        n_estimators: int = 8,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        """Init."""
        self.config_fp = config_fp
        # Store hyperparams
        self.n_estimators = n_estimators
        self.device = device
        self.kwargs = kwargs
        self.model: TabPFNClassifier | None = None

    def fit(self, df: pl.DataFrame) -> pd.DataFrame:
        """Fit."""
        # TODO: figure out how to free GPU memory
        # 1. Read config
        config = self._read_config()
        # Full training stage. No hyperparameter tuning
        # 2. Sort the df by EXPERIMENT, FRAME. Then compute bout_id
        df = df.sort([EXPERIMENT, FRAME])
        df = label_bouts(df)
        # 3. Resample
        resampler = RandomUnderSampler(sampling_strategy="auto", random_state=42)
        sub_df = df_resample(df, resampler)
        # 4. Init classifier
        self.model = TabPFNClassifier(
            n_estimators=self.n_estimators,
            device=self.device,
            random_state=config.seed,
            fit_mode="fit_with_cache",
            ignore_pretraining_limits=True,
            **self.kwargs,
        )
        # 6. Fit classifier on sub_df
        self.model.fit(df_get_features(sub_df), df_get_labels(sub_df))
        # 7. Set pcutoff as hardcoded 0.5 (tabpfn sorts itself out)
        config.pcutoff = 0.5
        self._write_config(config)
        # Return
        return pd.DataFrame(columns=pd.Index(["loss", "vloss"]))

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return predicted probabilities + binary preds with smoothing."""
        # Check model exists
        if self.model is None:
            msg = "model not yet trained."
            raise ValueError(msg)
        # Get configs
        config = self._read_config()
        # Predict
        frame = df.get_column(FRAME)
        prob = self.model.predict_proba(df_get_features(df))[:, 1]
        # Construct df
        y_df = pl.DataFrame(
            {
                FRAME: frame,
                BEHAVIOUR: config.behaviour_name,
                PROB: prob,
                PRED: prob > config.pcutoff,
            },
            schema=BEHAVIOUR_PREDICTED_SCHEMA,
        )
        # Smooth and return
        return smooth_preds(y_df, config.smoothing_frames, "median")

    def save(self) -> None:
        """Save."""
        # Check model exists
        if self.model is None:
            msg = "model not yet trained."
            raise ValueError(msg)
        # Save model
        model_dir = self.config_fp.parent
        save_fitted_tabpfn_model(self.model, model_dir / "model.tabpfn_fit")

    @classmethod
    def load(cls, config_fp: Path) -> Self:
        """Load."""
        model_dir = config_fp.parent
        # Instatiate
        inst = cls(config_fp=config_fp)
        # Load model
        inst.model = load_fitted_tabpfn_model(
            model_dir / "model.tabpfn_fit", device=get_gpu_device()
        )
        # Return
        return inst


class TorchAdapter(BaseAdapter):
    """Adapter for PyTorch models with lazy architecture instantiation."""

    framework: ClassVar[str] = "torch"

    def __init__(self, config_fp: Path, model: TorchModel) -> None:
        """Init."""
        self.model: TorchModel = model
        self.config_fp: Path = config_fp
        self.scaler: MinMaxScaler = MinMaxScaler()
        self.feature_mask: np.ndarray = np.ndarray([])

    def fit(self, df: pl.DataFrame) -> pd.DataFrame:
        """Fit."""
        config = self._read_config()
        # Preprocess
        x = self.scaler.fit_transform(df_get_features(df))
        self.feature_mask = select_features(
            x, df_get_labels(df), config.variance_threshold, config.max_features
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

    def _raw_predict(self, df: pl.DataFrame) -> tuple[pl.Series, Array1D]:
        x = df.drop(_META_COLS, strict=False).to_numpy()
        frame = df.get_column(FRAME)
        x = self.scaler.transform(x)[:, self.feature_mask]
        prob = self.model.predict(x, None, batch_size=256)
        return frame, prob

    def save(self) -> None:
        """Save."""
        if self.model is None or self.scaler is None or self.feature_mask is None:
            msg = "Cannot save unfitted torch model."
            raise RuntimeError(msg)

        model_dir = self.config_fp.parent
        torch.save(self.model.state_dict(), model_dir / "model.pt")
        joblib.dump(self.scaler, model_dir / "scaler.joblib")
        np.save(model_dir / "feature_mask.npy", self.feature_mask)

    @classmethod
    def load(cls, config_fp: Path) -> Self:
        """Reconstruct model + scaler + mask from version_dir artifacts.

        Requires self.model_factory to be set (from MODEL_REGISTRY
        instantiation before calling this method).
        """
        model_dir = config_fp.parent
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
        return inst


# ── registry ─────────────────────────────────────────────────────────


MODEL_TYPES_TO_CLASS: dict[str, type[BaseAdapter]] = {
    "sklearn": SklearnAdapter,
    "xgboost": XgboostAdapter,
    "tabpfn": TabPFNAdapter,
    "torch": TorchAdapter,
}

MODEL_TYPES_TO_STRING: dict[type[BaseAdapter], str] = {
    v: k for k, v in MODEL_TYPES_TO_CLASS.items()
}
