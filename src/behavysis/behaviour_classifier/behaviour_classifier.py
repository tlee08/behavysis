"""Behavioural classifier for training and inference on animal behaviour data."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from behavysis.constants import (
    BEHAVIOUR,
    BEHAVIOUR_SCORED_DIR,
    FEATURES_EXTRACTED_DIR,
    OUTCOME,
    PRED,
    PROB,
)

from .config import BehaviourClassifierConfig
from .data import (
    align_features_labels,
    load_feature_names,
    load_features,
    load_labels,
    stratified_split_by_video,
)
from .evaluation import (
    save_evaluation_results,
    save_feature_importance,
    save_feature_report,
    save_shap_summary,
    save_training_history,
)
from .registry import MODEL_REGISTRY
from .storage import classifier_fp, config_fp, eval_dir

if TYPE_CHECKING:
    from behavysis.pipeline.project import Project


class BehaviourClassifier:
    """Behavioural classifier — training, evaluation, and inference.

    Each instance is bound to one (project, behaviour_name) pair.
    The model type is determined by ``config.model_type``.

    Create a new (untrained) classifier::

        clf = BehaviourClassifier.create(proj_dir, "attack", config)
        clf.train()

    Load a trained classifier::

        clf = BehaviourClassifier.load(proj_dir, "attack")
        clf = BehaviourClassifier.load(proj_dir, "attack", model_type="dnn1")
    """

    def __init__(
        self,
        proj_dir: Path,
        behaviour_name: str,
        config: BehaviourClassifierConfig,
        adapter: object,
    ) -> None:
        self._proj_dir = proj_dir.resolve()
        self._behaviour_name = behaviour_name
        self._config = config
        self._adapter = adapter

    @property
    def config(self) -> BehaviourClassifierConfig:
        return self._config

    # ── factories ──────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        proj_dir: Path,
        behaviour_name: str,
        config: BehaviourClassifierConfig,
    ) -> BehaviourClassifier:
        """Create a new untrained classifier and persist its config.

        Writes ``config.yaml`` to disk immediately.
        """
        proj_dir = proj_dir.resolve()

        factory = MODEL_REGISTRY[config.model_type]
        adapter = factory()

        instance = cls(proj_dir, behaviour_name, config, adapter)
        instance._save_config()
        return instance

    @classmethod
    def create_all_from_project_dir(
        cls,
        proj_dir: Path,
        *,
        config: BehaviourClassifierConfig | None = None,
    ) -> list[BehaviourClassifier]:
        """Create classifiers for all labelled behaviours in a project."""
        proj_dir = proj_dir.resolve()
        from .data import list_behaviours

        behaviour_names = list_behaviours(proj_dir / BEHAVIOUR_SCORED_DIR)
        results = []
        for behav in behaviour_names:
            cfg = (
                config.model_copy(update={"behaviour_name": behav})
                if config
                else BehaviourClassifierConfig(behaviour_name=behav)
            )
            results.append(cls.create(proj_dir, behav, cfg))
        return results

    @classmethod
    def create_from_project(
        cls,
        proj: Project,
        *,
        config: BehaviourClassifierConfig | None = None,
    ) -> list[BehaviourClassifier]:
        """Create classifiers from a Project instance."""
        return cls.create_all_from_project_dir(proj.root_dir, config=config)

    @classmethod
    def load(
        cls,
        proj_dir: Path,
        behaviour_name: str,
        *,
        model_type: str | None = None,
    ) -> BehaviourClassifier:
        """Load a trained classifier from disk.

        Reads ``config.yaml`` and ``{model_type}/model.sav``.
        If ``model_type`` is omitted, uses the value from config.yaml.
        """
        proj_dir = proj_dir.resolve()
        fp = config_fp(proj_dir, behaviour_name)
        if not fp.exists():
            msg = (
                f'Model for behaviour "{behaviour_name}" not found in '
                f'"{proj_dir}". Train first or check path.'
            )
            raise FileNotFoundError(msg)

        config = BehaviourClassifierConfig.read_yaml(fp)
        mt = model_type or config.model_type

        model_fp = classifier_fp(proj_dir, behaviour_name, mt)
        if not model_fp.exists():
            available = [
                d.name for d in (config_fp(proj_dir, behaviour_name).parent).iterdir()
                if d.is_dir() and (d / "model.sav").exists()
            ]
            msg = (
                f'No trained model found for type "{mt}" in behaviour '
                f'"{behaviour_name}". Available: {available or "none"}'
            )
            raise FileNotFoundError(msg)

        from joblib import load

        adapter = load(model_fp)
        return cls(proj_dir, behaviour_name, config, adapter)

    # ── persistence ────────────────────────────────────────────────────

    def _save_config(self) -> None:
        fp = config_fp(self._proj_dir, self._behaviour_name)
        self._config.write_yaml(fp)

    def _save_model(self) -> None:
        fp = classifier_fp(
            self._proj_dir, self._behaviour_name, self._config.model_type,
        )
        self._adapter.save(fp)

    # ── training ───────────────────────────────────────────────────────

    def train(self) -> None:
        """Train the classifier and persist all artifacts."""
        logger.info(
            f"Training {self._config.model_type} for {self._behaviour_name}",
        )

        x_ls, x_names = load_features(self._proj_dir / FEATURES_EXTRACTED_DIR)
        y_ls, y_names = load_labels(
            self._proj_dir / BEHAVIOUR_SCORED_DIR,
            self._behaviour_name,
        )
        x_ls, y_ls, exp_names = align_features_labels(x_ls, y_ls, x_names, y_names)

        train_idx, test_idx = stratified_split_by_video(
            x_ls,
            y_ls,
            self._config.test_split,
            self._config.seed,
        )

        history = self._adapter.fit(x_ls, y_ls, train_idx, self._config)

        eval_d = eval_dir(
            self._proj_dir, self._behaviour_name, self._config.model_type,
        )
        save_training_history(history, eval_d)

        self._evaluate(x_ls, y_ls, train_idx, "train")
        self._evaluate(x_ls, y_ls, test_idx, "test")

        self._run_diagnostics(x_ls, eval_d)

        self._save_config()
        self._save_model()

        from .snapshot import TrainingSnapshot

        TrainingSnapshot.create(
            x_ls, y_ls, exp_names, self._proj_dir, self._config,
        )

    def _run_diagnostics(
        self,
        x_ls: list[np.ndarray],
        eval_d: Path,
    ) -> None:
        """Run diagnostic reporting: feature importance, SHAP, feature report."""
        feature_names = load_feature_names(self._proj_dir / FEATURES_EXTRACTED_DIR)
        if not feature_names:
            logger.warning("No feature names found for diagnostics.")
            return

        importances = self._get_feature_importances()

        save_feature_importance(feature_names, importances, eval_d)
        save_feature_report(feature_names, importances, eval_d)

        x_sample = np.concatenate([x[:100] for x in x_ls], axis=0)
        save_shap_summary(x_sample, feature_names, eval_d)

    def _get_feature_importances(self) -> np.ndarray:
        """Extract feature importances from the fitted adapter."""
        adapter = self._adapter
        n_features = adapter.scaler.n_features_in_
        importances = np.zeros(n_features, dtype=np.float64)

        if hasattr(adapter, "estimator"):
            est = adapter.estimator
            if hasattr(est, "feature_importances_"):
                importances = est.feature_importances_
            elif hasattr(est, "coef_"):
                importances = np.abs(est.coef_).flatten()

        return importances

    def _evaluate(
        self,
        x_ls: list[np.ndarray],
        y_ls: list[np.ndarray],
        index_ls: list[np.ndarray],
        name: str,
    ) -> None:
        y_true_ls = [y[idx] for y, idx in zip(y_ls, index_ls, strict=True)]
        y_prob_ls = [
            self._adapter.predict(x, idx, self._config.batch_size)
            for x, idx in zip(x_ls, index_ls, strict=True)
        ]

        y_true = np.concatenate(y_true_ls)
        y_prob = np.concatenate(y_prob_ls)
        y_pred = (y_prob > self._config.pcutoff).astype(int)

        d = eval_dir(
            self._proj_dir, self._behaviour_name, self._config.model_type,
        )
        save_evaluation_results(
            y_true,
            y_prob,
            y_pred,
            self._config.behaviour_name,
            self._config.pcutoff,
            d,
            name,
            index_ls,
        )

    # ── inference ─────────────────────────────────────────────────────

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Run inference on feature dataframe."""
        index = features_df.index
        x = features_df.to_numpy()
        y_prob = self._adapter.predict(
            x, np.arange(x.shape[0]), self._config.batch_size,
        )
        y_pred = (y_prob > self._config.pcutoff).astype(int)

        columns = pd.MultiIndex.from_tuples(
            [(self._config.behaviour_name, PROB), (self._config.behaviour_name, PRED)],
            names=[BEHAVIOUR, OUTCOME],
        )
        return pd.DataFrame(
            np.column_stack([y_prob, y_pred]),
            index=pd.Index(index, name="frame"),
            columns=columns,
        )


# ── batch training across model types ────────────────────────────────


def train_all_models(
    proj_dir: Path,
    behaviour_name: str,
    *,
    model_types: list[str] | None = None,
    config_overrides: dict | None = None,
) -> list[BehaviourClassifier]:
    """Train every model type in MODEL_REGISTRY for a behaviour.

    Each model's config and artifacts are saved. The final config.yaml
    reflects the last-trained model type as the active one.
    """
    types = model_types or list(MODEL_REGISTRY.keys())
    results = []
    for mt in types:
        config = BehaviourClassifierConfig(
            behaviour_name=behaviour_name,
            model_type=mt,
            **(config_overrides or {}),
        )
        clf = BehaviourClassifier.create(proj_dir, behaviour_name, config)
        clf.train()
        results.append(clf)
    return results
