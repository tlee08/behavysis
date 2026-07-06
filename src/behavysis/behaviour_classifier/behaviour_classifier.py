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
    load_features,
    load_labels,
    stratified_split_by_video,
)
from .evaluation import save_evaluation_results, save_training_history
from .registry import MODEL_REGISTRY
from .storage import classifier_fp, config_fp, eval_dir

if TYPE_CHECKING:
    from behavysis.pipeline.project import Project


class BehaviourClassifier:
    """Behavioural classifier — training, evaluation, and inference.

    Each instance is bound to one (project, behaviour_name) pair. The
    model type is determined by config.model_type, resolved through
    MODEL_REGISTRY.
    """

    def __init__(
        self,
        proj_dir: Path,
        behaviour_name: str,
        config: BehaviourClassifierConfig | None = None,
    ) -> None:
        self._proj_dir = proj_dir.resolve()
        self._behaviour_name = behaviour_name

        if config is None:
            config = self._load_or_create_config()
        else:
            config.proj_dir = self._proj_dir
            config.behaviour_name = self._behaviour_name
        self._config = config

        self._adapter = self._load_or_create_adapter()

    @property
    def config(self) -> BehaviourClassifierConfig:
        return self._config

    # ── factories ──────────────────────────────────────────────────────

    @classmethod
    def from_adapter(
        cls,
        proj_dir: Path,
        behaviour_name: str,
        config: BehaviourClassifierConfig,
    ) -> BehaviourClassifier:
        """Create with explicit config."""
        instance = cls.__new__(cls)
        instance._proj_dir = proj_dir.resolve()
        instance._behaviour_name = behaviour_name
        instance._config = config
        instance._adapter = instance._load_or_create_adapter()
        return instance

    @classmethod
    def create_all_from_project_dir(
        cls,
        proj_dir: Path,
    ) -> list[BehaviourClassifier]:
        """Create classifiers for all behaviours in a project directory."""
        proj_dir = proj_dir.resolve()
        from .data import list_behaviours

        behaviour_names = list_behaviours(proj_dir / BEHAVIOUR_SCORED_DIR)
        return [cls(proj_dir, behav) for behav in behaviour_names]

    @classmethod
    def create_from_project(cls, proj: Project) -> list[BehaviourClassifier]:
        """Create classifiers from Project instance."""
        return cls.create_all_from_project_dir(proj.root_dir)

    @classmethod
    def load(cls, proj_dir: Path, behaviour_name: str) -> BehaviourClassifier:
        """Load existing classifier from disk."""
        proj_dir = proj_dir.resolve()
        fp = config_fp(proj_dir, behaviour_name)
        if not fp.exists():
            msg = (
                f'Model for behaviour "{behaviour_name}" not found in "{proj_dir}". '
                "Check path or train first."
            )
            raise ValueError(msg)
        config = BehaviourClassifierConfig.model_validate_json(fp.read_text())
        return cls.from_adapter(proj_dir, behaviour_name, config)

    # ── config / adapter lifecycle ─────────────────────────────────────

    def _load_or_create_config(self) -> BehaviourClassifierConfig:
        fp = config_fp(self._proj_dir, self._behaviour_name)
        if fp.exists():
            logger.debug("Loaded existing config")
            return BehaviourClassifierConfig.model_validate_json(fp.read_text())
        logger.debug("Created new model config")
        return BehaviourClassifierConfig(
            proj_dir=self._proj_dir,
            behaviour_name=self._behaviour_name,
        )

    def _load_or_create_adapter(self):
        fp = classifier_fp(
            self._proj_dir,
            self._behaviour_name,
            self._config.model_type,
        )
        if fp.exists():
            logger.debug(f"Loaded existing adapter: {self._config.model_type}")
            from joblib import load

            return load(fp)

        logger.debug(f"Creating new adapter: {self._config.model_type}")
        factory = MODEL_REGISTRY[self._config.model_type]
        return factory()

    def _save(self) -> None:
        fp = classifier_fp(
            self._proj_dir, self._behaviour_name, self._config.model_type
        )
        self._adapter.save(fp)

    def _save_config(self) -> None:
        fp = config_fp(self._proj_dir, self._behaviour_name)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(self._config.model_dump_json(indent=2))

    # ── training ───────────────────────────────────────────────────────

    def train(self) -> None:
        """Train the classifier and save all artifacts."""
        logger.info(f"Training {self._config.model_type} for {self._behaviour_name}")

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

        eval_d = eval_dir(self._proj_dir, self._behaviour_name, self._config.model_type)
        save_training_history(history, eval_d)

        self._evaluate(x_ls, y_ls, train_idx, "train")
        self._evaluate(x_ls, y_ls, test_idx, "test")

        self._save_config()
        self._save()

        from .snapshot import TrainingSnapshot

        TrainingSnapshot.create(x_ls, y_ls, exp_names, self._config)

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

        d = eval_dir(self._proj_dir, self._behaviour_name, self._config.model_type)
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
        """Run inference on feature dataframe.

        Parameters
        ----------
        features_df : pd.DataFrame
            Unprocessed features DataFrame (wide format, index = frame).

        Returns:
        -------
        pd.DataFrame
            MultiIndex columns: (behaviour_name, prob), (behaviour_name, pred).
        """
        index = features_df.index
        x = features_df.to_numpy()
        y_prob = self._adapter.predict(
            x, np.arange(x.shape[0]), self._config.batch_size
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
    """Train every model in MODEL_REGISTRY (or the given subset) for a behaviour.

    Returns the list of trained BehaviourClassifier instances.
    """
    types = model_types or list(MODEL_REGISTRY.keys())
    results = []
    for model_type in types:
        config = BehaviourClassifierConfig(
            model_type=model_type,
            **(config_overrides or {}),
        )
        clf = BehaviourClassifier.from_adapter(proj_dir, behaviour_name, config)
        clf.train()
        results.append(clf)
    return results
