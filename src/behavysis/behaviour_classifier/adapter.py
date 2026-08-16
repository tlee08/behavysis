"""Model adapters for sklearn and PyTorch classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from tabpfn import TabPFNClassifier, load_fitted_tabpfn_model, save_fitted_tabpfn_model
from xgboost import XGBClassifier

from behavysis.constants import (
    BEHAVIOUR,
    EXPERIMENT,
    FRAME,
    PRED,
    PROB,
)
from behavysis.schemas import (
    BEHAVIOUR_BATCHED_PREDICTED_SCHEMA,
    BEHAVIOUR_PREDICTED_SCHEMA,
)
from behavysis.transforms import smooth_pred_bout, smooth_prob
from behavysis.utils import get_gpu_device

from .config import ModelRecipe
from .data import (
    ACTUAL,
    agg_eval_df_by_bouts,
    df_get_features,
    df_get_labels,
)
from .torch.architectures import MODEL_TYPES

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Self

    from sklearn.model_selection._search import BaseSearchCV

    from .torch.base import TorchModel


# -- post-processing optimisation ------------------------------------------

_SMOOTHING_GRID = (0, 1, 2, 3, 5, 8, 12)
_MIN_GAP_GRID = (0, 2, 4, 8, 16, 32)
_MIN_BOUT_GRID = (0, 2, 4, 8, 16, 32)
_N_PCUTOFF = 20
_MIN_PCUTOFF = 1e-6


def _best_pcutoff(
    smoothed: pl.DataFrame,
    min_gap: int,
    min_bout: int,
    pcutoffs: np.ndarray,
    target_recall: float,
) -> tuple[float, float]:
    """Best pcutoff (max precision at bout recall >= target) for one morphology."""
    best_pcutoff = float(pcutoffs[0])
    best_precision = -1.0
    for pcutoff in pcutoffs:
        pred_df = smoothed.with_columns(
            (pl.col(PROB) > pcutoff).cast(pl.Int64).alias(PRED)
        )
        pred_df = smooth_pred_bout(pred_df, min_gap=min_gap, min_bout=min_bout)
        bouts_df = agg_eval_df_by_bouts(pred_df)
        precision = precision_score(bouts_df[ACTUAL], bouts_df[PRED], zero_division=0)
        recall = recall_score(bouts_df[ACTUAL], bouts_df[PRED], zero_division=0)
        if recall >= target_recall and precision > best_precision:
            best_precision = precision
            best_pcutoff = float(pcutoff)
    return best_pcutoff, max(best_precision, 0.0)


class BaseAdapter(ABC):
    """Abstract adapter with fit / predict."""

    framework: ClassVar[str]
    recipe_fp: Path

    def _read_recipe(self) -> ModelRecipe:
        """Read recipe."""
        return ModelRecipe.read_yaml(self.recipe_fp)

    def _write_recipe(self, recipe: ModelRecipe) -> None:
        """Write recipe."""
        return recipe.write_yaml(self.recipe_fp)

    @abstractmethod
    def fit(self, df: pl.DataFrame) -> pd.DataFrame:
        """Train on rows where ``train_mask`` is True.

        Returns per-epoch history (empty for sklearn).
        """

    def optimise_postprocessing_parameters(self, val_df: pl.DataFrame) -> ModelRecipe:
        """Optimise smoothing/gap/bout/pcutoff on validation data.

        Sweeps the post-processing parameter grid and selects the combination
        that maximises bout-level precision subject to bout-level recall >=
        ``target_recall``.  Writes the result to the recipe and returns it.
        """
        recipe = self._read_recipe()
        raw = self.predict_raw(val_df).join(
            val_df.select([FRAME, EXPERIMENT, ACTUAL]),
            on=[FRAME, EXPERIMENT],
            how="left",
        )
        pcutoffs = np.unique(
            np.quantile(
                raw.get_column(PROB).to_numpy(),
                np.linspace(0.0, 1.0, _N_PCUTOFF),
            )
        )
        best_precision = -1.0
        best = (0, 0, 0, 0.0)
        for smoothing_frames in _SMOOTHING_GRID:
            smoothed = smooth_prob(
                raw, smoothing_frames=smoothing_frames, agg_func="median"
            )
            for min_gap in _MIN_GAP_GRID:
                for min_bout in _MIN_BOUT_GRID:
                    pcutoff, precision = _best_pcutoff(
                        smoothed, min_gap, min_bout, pcutoffs, recipe.target_recall
                    )
                    if precision > best_precision:
                        best_precision = precision
                        best = (smoothing_frames, min_gap, min_bout, pcutoff)
        smoothing_frames, min_gap, min_bout, pcutoff = best
        recipe.smoothing_frames = smoothing_frames
        recipe.min_gap_frames = min_gap
        recipe.min_bout_frames = min_bout
        recipe.pcutoff = max(float(pcutoff), _MIN_PCUTOFF)
        self._write_recipe(recipe)
        return recipe

    @abstractmethod
    def predict_raw(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return raw per-frame probabilities (frame, behaviour, prob, experiment).

        No smoothing, thresholding or bout morphology.
        """

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return predicted probabilities + binary preds with smoothing."""
        return self._predict_postprocess(self.predict_raw(df))

    def _predict_postprocess(self, raw_df: pl.DataFrame) -> pl.DataFrame:
        """Smooth, threshold and merge bouts on a raw prediction frame."""
        recipe = self._read_recipe()
        # Smooth frames with median filter
        df = smooth_prob(
            raw_df, smoothing_frames=recipe.smoothing_frames, agg_func="median"
        )
        # Make pred from prob cutoff
        df = df.with_columns((pl.col(PROB) > recipe.pcutoff).cast(pl.Int64).alias(PRED))
        # Smooth bouts by merging 3-frames-close, then dropping 3-frames large
        df = smooth_pred_bout(
            df, min_gap=recipe.min_gap_frames, min_bout=recipe.min_bout_frames
        )
        # Return
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

    def __init__(self, recipe_fp: Path, search: BaseSearchCV) -> None:
        """Init."""
        self.recipe_fp = recipe_fp
        self.search = search
        self.model: Pipeline | None = None

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
        if self.model is None:
            msg = "model not yet trained."
            raise ValueError(msg)
        recipe = self._read_recipe()
        raw_df = pl.DataFrame(
            {
                FRAME: df.get_column(FRAME),
                BEHAVIOUR: recipe.behaviour_name,
                PROB: pl.Series(self.model.predict_proba(df_get_features(df))[:, 1]),
            }
        )
        if EXPERIMENT in df.columns:
            raw_df = raw_df.with_columns(df.get_column(EXPERIMENT).alias(EXPERIMENT))
        return raw_df

    def save(self) -> None:
        """Save."""
        if self.model is None:
            msg = "model not yet trained."
            raise ValueError(msg)
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
        if self.model is None:
            msg = "model not yet trained."
            raise ValueError(msg)
        # clf is XGBoost, must first move to CPU before serialising
        preprocess: Pipeline = self.model[:-1]
        clf: XGBClassifier = self.model.steps[-1][1]
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
        inst = cls(recipe_fp, joblib.load(recipe_fp.with_name("search.joblib")))
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

    def __init__(self, recipe_fp: Path, **kwargs) -> None:  # noqa: ANN003
        """Init."""
        self.recipe_fp = recipe_fp
        # Store hyperparams
        self.kwargs = kwargs
        self.model: TabPFNClassifier | None = None

    def fit(self, df: pl.DataFrame) -> pd.DataFrame:
        """Fit."""
        self.model = TabPFNClassifier(**self.kwargs)
        self.model.fit(df_get_features(df), df_get_labels(df))
        return pd.DataFrame(columns=pd.Index(["loss", "vloss"]))

    def predict_raw(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return raw per-frame probabilities (frame, behaviour, prob, experiment)."""
        if self.model is None:
            msg = "model not yet trained."
            raise ValueError(msg)
        recipe = self._read_recipe()
        raw_df = pl.DataFrame(
            {
                FRAME: df.get_column(FRAME),
                BEHAVIOUR: recipe.behaviour_name,
                PROB: pl.Series(self.model.predict_proba(df_get_features(df))[:, 1]),
            }
        )
        if EXPERIMENT in df.columns:
            raw_df = raw_df.with_columns(df.get_column(EXPERIMENT).alias(EXPERIMENT))
        return raw_df

    def save(self) -> None:
        """Save."""
        if self.model is None:
            msg = "model not yet trained."
            raise ValueError(msg)
        # Save model
        self.recipe_fp.parent.mkdir(parents=True, exist_ok=True)
        save_fitted_tabpfn_model(
            self.model, self.recipe_fp.with_name("model.tabpfn_fit")
        )

    @classmethod
    def load(cls, recipe_fp: Path) -> Self:
        """Load."""
        # Instatiate
        inst = cls(recipe_fp)
        # Load model
        inst.model = load_fitted_tabpfn_model(
            recipe_fp.with_name("model.tabpfn_fit"), device=get_gpu_device()
        )
        # Return
        return inst


class TorchAdapter(BaseAdapter):
    """Adapter for PyTorch sequence models (1D temporal CNN)."""

    framework: ClassVar[str] = "torch"

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
        self.model: TorchModel | None = None
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
        if self.model is None:
            msg = "model not yet trained."
            raise ValueError(msg)
        recipe = self._read_recipe()
        df = (
            df.sort([EXPERIMENT, FRAME]) if EXPERIMENT in df.columns else df.sort(FRAME)
        )
        x_ls = [self._transform(x) for x in self._to_x_ls(df)]
        prob = self.model.predict(x_ls, batch_size=self.batch_size)
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
        if self.model is None:
            msg = "model not yet trained."
            raise ValueError(msg)
        self.recipe_fp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": self.model_cls.__name__,
                "window_frames": self.window_frames,
                "state_dict": self.model.state_dict(),
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

MODEL_TYPES_TO_STRING: dict[type[BaseAdapter], str] = {
    v: k for k, v in MODEL_TYPES_TO_CLASS.items()
}
