"""Model adapters for sklearn and PyTorch classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from tabpfn import TabPFNClassifier, load_fitted_tabpfn_model, save_fitted_tabpfn_model
from xgboost import XGBClassifier

from behavysis.constants import BEHAVIOUR, EXPERIMENT, FRAME, PRED, PROB
from behavysis.schemas import (
    BEHAVIOUR_BATCHED_PREDICTED_SCHEMA,
    BEHAVIOUR_PREDICTED_SCHEMA,
)
from behavysis.transforms import hysteresis, smooth_pred_bout, smooth_prob
from behavysis.utils import get_gpu_device

from .config import FRAME_AWARE, HYSTERESIS, ModelRecipe
from .data import ACTUAL, df_get_features, df_get_labels
from .parameter_optimisation import optimise_postprocessing
from .torch.architectures import MODEL_TYPES

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Self

    from sklearn.model_selection._search import BaseSearchCV

    from .torch.base import TorchModel


class BaseAdapter(ABC):
    """Abstract adapter with fit / predict."""

    framework: ClassVar[str]
    recipe_fp: Path
    model: object | None

    def _read_recipe(self) -> ModelRecipe:
        """Read recipe."""
        return ModelRecipe.read_yaml(self.recipe_fp)

    def _write_recipe(self, recipe: ModelRecipe) -> None:
        """Write recipe."""
        return recipe.write_yaml(self.recipe_fp)

    def _check_model_trained(self) -> None:
        if self.model is None:
            msg = "model not yet trained."
            raise ValueError(msg)

    @abstractmethod
    def fit(self, df: pl.DataFrame) -> pd.DataFrame:
        """Train on rows where ``train_mask`` is True.

        Returns per-epoch history (empty for sklearn).
        """

    def optimise_postprocessing(self, val_df: pl.DataFrame) -> ModelRecipe:
        """Optimise post-processing parameters on validation data.

        Selects the parameters of ``recipe.postprocessing_step`` that minimise the
        review cost (hidden events first, then predicted bouts) subject to bout-level
        recall >= ``target_recall``. Writes the result to the recipe and returns it.
        """
        recipe = self._read_recipe()
        raw = self.predict_raw(val_df).join(
            val_df.select([FRAME, EXPERIMENT, ACTUAL]),
            on=[FRAME, EXPERIMENT],
            how="left",
        )
        recipe = optimise_postprocessing(recipe, raw)
        self._write_recipe(recipe)
        return recipe

    @abstractmethod
    def predict_raw(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return raw per-frame probabilities (frame, behaviour, prob[, experiment]).

        No smoothing, thresholding or bout morphology.
        """

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return predicted probabilities + binary preds with smoothing."""
        return self._predict_postprocess(self.predict_raw(df))

    def _predict_postprocess(self, raw_df: pl.DataFrame) -> pl.DataFrame:
        """Apply the configured post-processing to a raw prediction frame."""
        recipe = self._read_recipe()
        if recipe.postprocessing_step == FRAME_AWARE:
            cfg = recipe.frame_aware
            df = smooth_prob(
                raw_df, smoothing_frames=cfg.smoothing_frames, agg_func="median"
            )
            df = df.with_columns(
                (pl.col(PROB) > cfg.pcutoff).cast(pl.Int64).alias(PRED)
            )
            df = smooth_pred_bout(
                df, min_gap=cfg.min_gap_frames, min_bout=cfg.min_bout_frames
            )
        elif recipe.postprocessing_step == HYSTERESIS:
            cfg = recipe.hysteresis
            df = hysteresis(
                raw_df, pcutoff=cfg.pcutoff, low_threshold=cfg.low_threshold
            )
            df = smooth_pred_bout(df, min_gap=0, min_bout=cfg.min_bout_frames)
        if EXPERIMENT in df.columns:
            return pl.DataFrame(
                df.select(list(BEHAVIOUR_BATCHED_PREDICTED_SCHEMA)),
                BEHAVIOUR_BATCHED_PREDICTED_SCHEMA,
            )
        return pl.DataFrame(
            df.select(list(BEHAVIOUR_PREDICTED_SCHEMA)),
            schema=BEHAVIOUR_PREDICTED_SCHEMA,
        )

    @abstractmethod
    def save(self) -> None:
        """Persist model artifacts inside dst_dir."""

    @classmethod
    @abstractmethod
    def load(cls, recipe_fp: Path) -> Self:
        """Load model artifacts."""


class SklearnAdapter(BaseAdapter):
    """Sklearn adapter."""

    framework: ClassVar[str] = "sklearn"
    model: Pipeline | None

    def __init__(self, recipe_fp: Path, search: BaseSearchCV) -> None:
        """Init."""
        self.recipe_fp = recipe_fp
        self.search = search
        self.model = None

    def fit(self, df: pl.DataFrame) -> pd.DataFrame:
        """Fit."""
        self.search.fit(
            df_get_features(df),
            df_get_labels(df),
            groups=df.get_column(EXPERIMENT).to_numpy(),
        )
        self.model = self.search.best_estimator_
        return pd.DataFrame(columns=pd.Index(["loss", "vloss"]))

    def predict_raw(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return raw per-frame probabilities (frame, behaviour, prob, experiment)."""
        self._check_model_trained()
        recipe = self._read_recipe()
        raw_df = pl.DataFrame(
            {
                FRAME: df.get_column(FRAME),
                BEHAVIOUR: recipe.behaviour_name,
                PROB: pl.Series(self.model.predict_proba(df_get_features(df))[:, 1]),  # ty: ignore[unresolved-attribute]
            }
        )
        if EXPERIMENT in df.columns:
            raw_df = raw_df.with_columns(df.get_column(EXPERIMENT).alias(EXPERIMENT))
        return raw_df

    def save(self) -> None:
        """Save."""
        self._check_model_trained()
        self.recipe_fp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.search, self.recipe_fp.with_name("search.joblib"))
        joblib.dump(self.model, self.recipe_fp.with_name("model.joblib"))

    @classmethod
    def load(cls, recipe_fp: Path) -> Self:
        """Load."""
        inst = cls(recipe_fp, joblib.load(recipe_fp.with_name("search.joblib")))
        inst.model = joblib.load(recipe_fp.with_name("model.joblib"))
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
        self._check_model_trained()
        # clf is XGBoost, must first move to CPU before serialising
        preprocess: Pipeline = self.model[:-1]  # ty: ignore[not-subscriptable]
        clf: XGBClassifier = self.model.steps[-1][1]  # ty: ignore[unresolved-attribute]
        # Save
        self.recipe_fp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.search, self.recipe_fp.with_name("search.joblib"))
        joblib.dump(preprocess, self.recipe_fp.with_name("preprocess.joblib"))
        clf.save_model(self.recipe_fp.with_name("clf.ubj"))

    @classmethod
    def load(cls, recipe_fp: Path) -> Self:
        """Load.

        Must load XGBoost model as a .ubj so it serialisable to all machines.
        """
        # Instatiate
        # Note: not loading search object, as ita) is not needed for inference
        # and b) xgboost can't be serialised across machines
        # search = joblib.load(recipe_fp.with_name("search.joblib"))  # noqa: ERA001
        search = GridSearchCV(Pipeline([("clf", RandomForestClassifier())]), {}, cv=3)
        inst = cls(recipe_fp, search)
        # Load pipelin
        preprocess: Pipeline = joblib.load(recipe_fp.with_name("preprocess.joblib"))
        # Load model
        clf = XGBClassifier(device=get_gpu_device())
        clf.load_model(recipe_fp.with_name("clf.ubj"))
        # Reconstruct pipeline
        inst.model = Pipeline([*preprocess.steps, ("clf", clf)])
        # Return
        return inst


class TabpfnAdapter(BaseAdapter):
    """Adapter for TabPFN."""

    framework: ClassVar[str] = "tabpfn"
    model: TabPFNClassifier | None

    def __init__(self, recipe_fp: Path, **kwargs) -> None:  # noqa: ANN003
        """Init."""
        self.recipe_fp = recipe_fp
        # Store hyperparams
        self.kwargs = kwargs
        self.model = None

    def fit(self, df: pl.DataFrame) -> pd.DataFrame:
        """Fit."""
        self.model = TabPFNClassifier(**self.kwargs)
        self.model.fit(df_get_features(df), df_get_labels(df))
        return pd.DataFrame(columns=pd.Index(["loss", "vloss"]))

    def predict_raw(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return raw per-frame probabilities (frame, behaviour, prob, experiment)."""
        self._check_model_trained()
        recipe = self._read_recipe()
        raw_df = pl.DataFrame(
            {
                FRAME: df.get_column(FRAME),
                BEHAVIOUR: recipe.behaviour_name,
                PROB: pl.Series(self.model.predict_proba(df_get_features(df))[:, 1]),  # ty: ignore[unresolved-attribute]
            }
        )
        if EXPERIMENT in df.columns:
            raw_df = raw_df.with_columns(df.get_column(EXPERIMENT).alias(EXPERIMENT))
        return raw_df

    def save(self) -> None:
        """Save."""
        self._check_model_trained()
        # Save model
        self.recipe_fp.parent.mkdir(parents=True, exist_ok=True)
        save_fitted_tabpfn_model(
            self.model,  # ty: ignore[invalid-argument-type]
            self.recipe_fp.with_name("model.tabpfn_fit"),
        )

    @classmethod
    def load(cls, recipe_fp: Path) -> Self:
        """Load."""
        # Instatiate
        inst = cls(recipe_fp)
        # Load model
        inst.model = load_fitted_tabpfn_model(  # ty: ignore[invalid-assignment]
            recipe_fp.with_name("model.tabpfn_fit"), device=get_gpu_device()
        )
        # Return
        return inst


class TorchAdapter(BaseAdapter):
    """Adapter for PyTorch sequence models (1D temporal CNN)."""

    framework: ClassVar[str] = "torch"
    model: TorchModel | None

    def __init__(
        self,
        recipe_fp: Path,
        model_cls: type[TorchModel],
        window_frames: int,
        batch_size: int = 256,
        epochs: int = 15,
    ) -> None:
        """Init."""
        self.recipe_fp = recipe_fp
        self.model_cls = model_cls
        self.window_frames = window_frames
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.feature_cols: list[str] = []
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, df: pl.DataFrame) -> pd.DataFrame:
        """Fit the model and calibrate pcutoff on held-out experiments."""
        # Per-experiment arrays, standardised with NaN -> mean imputation.
        self.feature_cols = df_get_features(df).columns
        x_ls, y_ls = self._to_arrays(df)
        self._fit_scaler(x_ls)
        x_ls = [self._transform(x) for x in x_ls]
        # Train.
        self.model = self.model_cls(len(self.feature_cols), self.window_frames)
        return self.model.fit(x_ls, y_ls, self.batch_size, self.epochs)

    def predict_raw(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return raw per-frame probabilities (frame, behaviour, prob, experiment)."""
        self._check_model_trained()
        recipe = self._read_recipe()
        df = (
            df.sort([EXPERIMENT, FRAME]) if EXPERIMENT in df.columns else df.sort(FRAME)
        )
        x_ls = [self._transform(x) for x in self._to_x_ls(df)]
        prob = self.model.predict(x_ls, batch_size=self.batch_size)  # ty: ignore[unresolved-attribute]
        raw_df = pl.DataFrame(
            {
                FRAME: df.get_column(FRAME),
                BEHAVIOUR: recipe.behaviour_name,
                PROB: pl.Series(prob),
            }
        )
        if EXPERIMENT in df.columns:
            raw_df = raw_df.with_columns(df.get_column(EXPERIMENT).alias(EXPERIMENT))
        return raw_df

    def save(self) -> None:
        """Save model state, scaler and feature metadata."""
        self._check_model_trained()
        self.recipe_fp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": self.model_cls.__name__,
                "window_frames": self.window_frames,
                "state_dict": self.model.state_dict(),  # ty: ignore[unresolved-attribute]
                "mean": self._mean,
                "std": self._std,
                "feature_cols": self.feature_cols,
                "batch_size": self.batch_size,
            },
            self.recipe_fp.with_name("model.pt"),
        )

    @classmethod
    def load(cls, recipe_fp: Path) -> Self:
        """Load model state, scaler and feature metadata."""
        payload = torch.load(recipe_fp.with_name("model.pt"), weights_only=False)
        model_cls = MODEL_TYPES[payload["model_type"]]
        inst = cls(
            recipe_fp,
            model_cls,
            payload["window_frames"],
            batch_size=payload["batch_size"],
        )
        inst._mean = payload["mean"]
        inst._std = payload["std"]
        inst.feature_cols = payload["feature_cols"]
        nfeatures = payload["mean"].shape[0]
        inst.model = model_cls(nfeatures, payload["window_frames"])
        inst.model.load_state_dict(payload["state_dict"])
        inst.model.eval()
        return inst

    # -- helpers -----------------------------------------------------------

    def _to_x_ls(self, df: pl.DataFrame) -> list[np.ndarray]:
        """Per-experiment feature matrices, sorted by frame."""
        parts = (
            df.partition_by([EXPERIMENT], maintain_order=True)
            if EXPERIMENT in df.columns
            else [df]
        )
        return [
            p.sort(FRAME).select(self.feature_cols).to_numpy().astype(np.float32)
            for p in parts
        ]

    def _to_arrays(self, df: pl.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Per-experiment feature matrices and label vectors."""
        parts = (
            df.partition_by([EXPERIMENT], maintain_order=True)
            if EXPERIMENT in df.columns
            else [df]
        )
        x_ls: list[np.ndarray] = []
        y_ls: list[np.ndarray] = []
        for p in parts:
            sub = p.sort(FRAME)
            x_ls.append(sub.select(self.feature_cols).to_numpy().astype(np.float32))
            y_ls.append(sub.get_column(ACTUAL).to_numpy().astype(np.float32))
        return x_ls, y_ls

    def _fit_scaler(self, x_ls: list[np.ndarray]) -> None:
        """Per-feature mean/std from the training data (NaN-ignoring)."""
        x_all = np.concatenate(x_ls, axis=0)
        self._mean = np.nanmean(x_all, axis=0)
        self._std = np.nanstd(x_all, axis=0)
        self._std[self._std == 0] = 1.0

    def _transform(self, x: np.ndarray) -> np.ndarray:
        """Impute NaN with the feature mean, then standardise."""
        x = np.where(np.isfinite(x), x, self._mean)
        return (x - self._mean) / self._std


# -- registry ---------------------------------------------------------


MODEL_TYPES_TO_CLASS: dict[str, type[BaseAdapter]] = {
    "sklearn": SklearnAdapter,
    "xgboost": XgboostAdapter,
    "tabpfn": TabpfnAdapter,
    "torch": TorchAdapter,
}
