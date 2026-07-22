"""Model adapters for sklearn and PyTorch classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
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
from behavysis.transforms import smooth_prob
from behavysis.transforms.behaviour import smooth_bouts
from behavysis.utils import get_gpu_device

from .config import ModelRecipe
from .data import (
    agg_eval_df_by_bouts,
    df_get_features,
    df_get_labels,
    label_bouts,
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

    @abstractmethod
    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return predicted probabilities + binary preds with smoothing."""

    def _predict_postprocess(
        self, prob: pl.Series, frame: pl.Series, experiment: pl.Series | None = None
    ) -> pl.DataFrame:
        # Get recipe
        recipe = self._read_recipe()
        # Construct df
        df = pl.DataFrame(
            {
                FRAME: frame,
                BEHAVIOUR: recipe.behaviour_name,
                PROB: prob,
                PRED: 0,  # placeholder
            },
            schema=BEHAVIOUR_PREDICTED_SCHEMA,
        )
        # Add experiment column if given
        if experiment:
            df = df.with_columns(experiment.alias(EXPERIMENT))
        # Smooth frames with median filter
        df = smooth_prob(
            df, smoothing_frames=recipe.smoothing_frames, agg_func="median"
        )
        # Make pred from prob cutoff
        df = df.with_columns((pl.col(PROB) > recipe.pcutoff).alias(PRED))
        # Smooth bouts by merging 3-frames-close, then dropping 3-frames large
        # Return
        return smooth_bouts(df, min_gap=recipe.min_gap, min_bout=recipe.min_bout)

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
        # 1. Read recipe
        recipe = self._read_recipe()
        # Hyperparameter selection stage
        # 2. Sort the df by EXPERIMENT, FRAME. Then compute bout_id
        df = df.sort([EXPERIMENT, FRAME])
        df = label_bouts(df, ACTUAL)
        # 4. hyperparameter selection on df. CV grouped-by-bout_id
        self.search.refit = False
        self.search.fit(
            df_get_features(df),
            df_get_labels(df),
            groups=df.get_column(BOUT_ID).to_numpy(),
        )
        # Full training stage
        # 5. Make train-val split by bouts (to programatically find best pcutoff)
        train_idx, val_idx = stratified_split_by_group(
            df, recipe.val_split, BOUT_ID, recipe.seed
        )
        train_df = df.gather(train_idx)
        # 6. Refit best pipeline on train_df
        self.model = clone(self.search.estimator).set_params(**self.search.best_params_)
        self.model.fit(df_get_features(train_df), df_get_labels(train_df))
        # 7. Find best pcutoff with val_idx and update recipe with best pcutoff value
        # Use per-bouts eval instead of per-frames eval
        y_df = self.predict(df).with_columns(
            df.get_column(ACTUAL),
            df.get_column(BOUT_ID),
        )
        y_val_df = y_df.gather(val_idx)
        y_val_bouts_df = agg_eval_df_by_bouts(y_val_df)
        _, recall, thresholds = precision_recall_curve(
            y_val_bouts_df.get_column(ACTUAL),
            y_val_bouts_df.get_column(PROB),
            drop_intermediate=True,
        )
        recipe.pcutoff = float(
            thresholds[(recall[:-1] >= recipe.target_recall)][-1]
            if np.any(recall[:-1] >= recipe.target_recall)
            else 0.001
        )
        self._write_recipe(recipe)
        # Return
        return pd.DataFrame(columns=pd.Index(["loss", "vloss"]))

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return predicted probabilities + binary preds with smoothing."""
        # Check model exists
        if self.model is None:
            msg = "model not yet trained."
            raise ValueError(msg)
        # Predict
        frame = df.get_column(FRAME)
        prob = pl.Series(self.model.predict_proba(df_get_features(df))[:, 1])
        # Get experiment column if it exists
        experiment = None
        if EXPERIMENT in df.columns:
            experiment = df.get_column(EXPERIMENT)
        # Postprocess and return
        return self._predict_postprocess(prob, frame, experiment)

    def save(self) -> None:
        """Save."""
        # Check model exists
        if self.model is None:
            msg = "model not yet trained."
            raise ValueError(msg)
        # Save
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
        # Check model exists
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

    def __init__(self, recipe_fp: Path, **kwargs) -> None:
        """Init."""
        self.recipe_fp = recipe_fp
        # Store hyperparams
        self.kwargs = kwargs
        self.model: TabPFNClassifier | None = None

    def fit(self, df: pl.DataFrame) -> pd.DataFrame:
        """Fit."""
        # 1. Read recipe
        recipe = self._read_recipe()
        # Full training stage. No hyperparameter tuning
        # 2. Sort the df by EXPERIMENT, FRAME. Then compute bout_id
        df = df.sort([EXPERIMENT, FRAME])
        df = label_bouts(df, ACTUAL)
        # 3. Init classifier
        self.model = TabPFNClassifier(
            **self.kwargs,
        )
        # 4. Fit classifier on sub_df
        self.model.fit(df_get_features(df), df_get_labels(df))
        # 5. Set pcutoff as hardcoded 0.5 (tabpfn sorts itself out)
        recipe.pcutoff = 0.5
        self._write_recipe(recipe)
        # Return
        return pd.DataFrame(columns=pd.Index(["loss", "vloss"]))

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return predicted probabilities + binary preds with smoothing."""
        # Check model exists
        if self.model is None:
            msg = "model not yet trained."
            raise ValueError(msg)
        # Predict
        frame = df.get_column(FRAME)
        prob = pl.Series(self.model.predict_proba(df_get_features(df))[:, 1])
        # Get experiment column if it exists
        experiment = None
        if EXPERIMENT in df.columns:
            experiment = df.get_column(EXPERIMENT)
        # Postprocess and return
        return self._predict_postprocess(prob, frame, experiment)

    def save(self) -> None:
        """Save."""
        # Check model exists
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
    """Adapter for PyTorch models with lazy architecture instantiation."""

    framework: ClassVar[str] = "torch"

    def __init__(self, recipe_fp: Path, model: TorchModel) -> None:
        """Init."""
        self.model: TorchModel = model
        self.recipe_fp: Path = recipe_fp
        self.scaler: MinMaxScaler = MinMaxScaler()
        self.feature_mask: np.ndarray = np.ndarray([])

    def fit(self, df: pl.DataFrame) -> pd.DataFrame:
        """Fit."""
        recipe = self._read_recipe()
        # Preprocess
        x = self.scaler.fit_transform(df_get_features(df))
        self.feature_mask = select_features(
            x, df_get_labels(df), recipe.variance_threshold, recipe.max_features
        )
        nfeatures = len(self.feature_mask)
        # Train
        return self.model.fit(
            [x[:, self.feature_mask]],
            [y],
            [np.arange(x.shape[0])],
            batch_size=recipe.batch_size,
            epochs=recipe.epochs,
            val_split=recipe.val_split,
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

        model_dir = self.recipe_fp.parent
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.recipe_fp.with_name("model.pt"))
        joblib.dump(self.scaler, self.recipe_fp.with_name("scaler.joblib"))
        np.save(self.recipe_fp.with_name("feature_mask.npy"), self.feature_mask)

    @classmethod
    def load(cls, recipe_fp: Path) -> Self:
        """Reconstruct model + scaler + mask from version_dir artifacts.

        Requires self.model_factory to be set (from MODEL_REGISTRY
        instantiation before calling this method).
        """
        model = model.load_state_dict(
            torch.load(
                recipe_fp.with_name("model.pt"),
                map_location=torch.device("cpu"),
                weights_only=True,
            )
        )
        inst = cls(recipe_fp)
        inst.scaler = joblib.load(recipe_fp.with_name("scaler.joblib"))
        inst.feature_mask = np.load(recipe_fp.with_name("feature_mask.npy"))
        return inst


# ── registry ─────────────────────────────────────────────────────────


MODEL_TYPES_TO_CLASS: dict[str, type[BaseAdapter]] = {
    "sklearn": SklearnAdapter,
    "xgboost": XgboostAdapter,
    "tabpfn": TabpfnAdapter,
    "torch": TorchAdapter,
}

MODEL_TYPES_TO_STRING: dict[type[BaseAdapter], str] = {
    v: k for k, v in MODEL_TYPES_TO_CLASS.items()
}
