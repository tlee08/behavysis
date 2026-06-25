"""Behavioral classifier for training and inference on animal behavior data."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from joblib import dump, load
from loguru import logger

from behavysis.behav_classifier.evaluation import (
    save_evaluation_results,
    save_training_history,
)
from behavysis.constants import PRED, PROB, Folders
from behavysis.df_classes import BehavPredictedDf
from behavysis.models import BehavClassifierConfig

from .clf_models.base_torch_model import BaseTorchModel
from .clf_models.clf_templates import CLF_TEMPLATES, CNN1
from .data import (
    combine_dfs,
    prepare_training_data,
    preproc_x_transform,
    wrangle_columns_y,
)

if TYPE_CHECKING:
    from behavysis.pipeline.project import Project


class BehavClassifier:
    """Behavioral classifier for training and inference.

    Manages model training, evaluation, and prediction for animal behavior
    classification from pose estimation features.

    Attributes:
    ----------
    proj_dir : Path
        Project directory containing features and scored behaviors.
    behav_name : str
        Name of the behavior being classified.
    clf : BaseTorchModel
        The trained classifier model.
    """

    _proj_dir: Path
    _behav_name: str
    _clf: BaseTorchModel

    def __init__(self, proj_dir: Path, behav_name: str) -> None:
        """Initialize classifier for a specific behavior.

        Parameters
        ----------
        proj_dir : Path
            Project directory path.
        behav_name : str
            Name of behavior to classify.

        Raises:
        ------
        AssertionError
            If behavior is not found in scored behaviors.
        """
        self._proj_dir = proj_dir.resolve()
        self._behav_name = behav_name

        # Verify behavior exists in scored data
        y_df = wrangle_columns_y(combine_dfs(self.y_dir))
        assert np.isin(behav_name, y_df.columns), (
            f"Behavior '{behav_name}' not found in scored behaviors"
        )

        # Load or create config
        try:
            config = BehavClassifierConfig.model_validate_json(
                self.config_fp.read_text()
            )
            logger.debug("Loaded existing config")
        except FileNotFoundError:
            config = BehavClassifierConfig()
            logger.debug("Created new model config")

        config.proj_dir = self._proj_dir
        config.behav_name = self._behav_name
        self.config = config

        # Load or create classifier
        try:
            self.clf = load(self.clf_fp)
            logger.debug("Loaded existing classifier")
        except FileNotFoundError:
            self.clf = CNN1()
            logger.debug("Created new classifier")

    #################################################
    # Properties
    #################################################

    @property
    def proj_dir(self) -> Path:
        """Project directory."""
        return self._proj_dir

    @property
    def behav_name(self) -> str:
        """Behavior name."""
        return self._behav_name

    @property
    def clf(self) -> BaseTorchModel:
        """Classifier model."""
        return self._clf

    @clf.setter
    def clf(self, clf: BaseTorchModel | str) -> None:
        """Set classifier from model instance or saved model name."""
        if isinstance(clf, str):
            clf_name = clf
            self._clf = load(self.clfs_dir / clf / "classifier.sav")
            logger.debug(f"Loaded classifier: {clf_name}")
        else:
            clf_name = type(clf).__name__
            self._clf = clf
            logger.debug(f"Initialized classifier: {clf_name}")

        config = self.config
        config.clf_struct = clf_name
        self.config = config

    @property
    def model_dir(self) -> Path:
        """Model directory for this behavior."""
        return self.proj_dir / "behav_models" / self.behav_name

    @property
    def config_fp(self) -> Path:
        """Path to model config file."""
        return self.model_dir / "config.json"

    @property
    def config(self) -> BehavClassifierConfig:
        """Current model configuration."""
        return BehavClassifierConfig.model_validate_json(self.config_fp.read_text())

    @config.setter
    def config(self, config: BehavClassifierConfig) -> None:
        """Update model configuration."""
        try:
            if self.config == config:
                return
        except FileNotFoundError:
            pass
        logger.debug("Config changed. Updating on disk")
        self.config_fp.write_text(config.model_dump_json(indent=2))

    @property
    def clfs_dir(self) -> Path:
        """Directory containing classifier variants."""
        return self.model_dir / "classifiers"

    @property
    def clf_dir(self) -> Path:
        """Directory for current classifier structure."""
        return self.clfs_dir / self.config.clf_struct

    @property
    def clf_fp(self) -> Path:
        """Path to saved classifier."""
        return self.clf_dir / "classifier.sav"

    @property
    def preproc_fp(self) -> Path:
        """Path to preprocessing pipeline."""
        return self.clf_dir / "preproc.sav"

    @property
    def eval_dir(self) -> Path:
        """Directory for evaluation outputs."""
        return self.clf_dir / "evaluation"

    @property
    def x_dir(self) -> Path:
        """Directory containing feature files."""
        return self.proj_dir / Folders.FEATURES_EXTRACTED.value

    @property
    def y_dir(self) -> Path:
        """Directory containing scored behavior files."""
        return self.proj_dir / Folders.SCORED_BEHAVS.value

    #################################################
    # Factory Methods
    #################################################

    @classmethod
    def create_from_project_dir(cls, proj_dir: Path) -> list[BehavClassifier]:
        """Create classifiers for all behaviors in project.

        Parameters
        ----------
        proj_dir : Path
            Project directory path.

        Returns:
        -------
        list[BehavClassifier]
            List of BehavClassifier instances, one per behavior.
        """
        proj_dir = proj_dir.resolve()
        y_df = wrangle_columns_y(combine_dfs(proj_dir / Folders.SCORED_BEHAVS.value))
        behavs_ls = y_df.columns.to_list()
        return [cls(proj_dir, behav) for behav in behavs_ls]

    @classmethod
    def create_from_project(cls, proj: Project) -> list[BehavClassifier]:
        """Create classifiers from Project instance.

        Parameters
        ----------
        proj : Project
            Behavysis Project instance.

        Returns:
        -------
        list[BehavClassifier]
            List of BehavClassifier instances.
        """
        return cls.create_from_project_dir(proj.root_dir)

    @classmethod
    def load(cls, proj_dir: Path, behav_name: str) -> BehavClassifier:
        """Load existing classifier.

        Parameters
        ----------
        proj_dir : Path
            Project directory path.
        behav_name : str
            Behavior name.

        Returns:
        -------
        BehavClassifier
            Loaded classifier instance.

        Raises:
        ------
        ValueError
            If model config file not found.
        """
        proj_dir = proj_dir.resolve()
        config_fp = proj_dir / "behav_models" / behav_name / "config.json"
        try:
            BehavClassifierConfig.model_validate_json(config_fp.read_text())
        except (FileNotFoundError, OSError) as e:
            msg = (
                f'Model in "{proj_dir}" with behavior "{behav_name}" not found. '
                "Check file path."
            )
            raise ValueError(msg) from e
        return cls(proj_dir, behav_name)

    #################################################
    # Training Pipeline
    #################################################

    def pipeline_training(self) -> None:
        """Train classifier and save to model directory."""
        logger.info(f"Training {self.config.clf_struct}")

        # Prepare data
        x_ls, y_ls, index_train_ls, index_test_ls = prepare_training_data(
            self.x_dir,
            self.y_dir,
            self.config.behav_name,
            self.preproc_fp,
            self.config.test_split,
            self.config.oversample_ratio,
            self.config.undersample_ratio,
        )

        # Train model
        history = self.clf.fit(
            x_ls=x_ls,
            y_ls=y_ls,
            index_ls=index_train_ls,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            val_split=self.config.val_split,
        )

        # Save training history
        save_training_history(history, self.eval_dir)

        # Evaluate on train and test sets
        self._evaluate_and_save(x_ls, y_ls, index_train_ls, "train")
        self._evaluate_and_save(x_ls, y_ls, index_test_ls, "test")

        # Save model
        dump(self.clf, self.clf_fp)

    def pipeline_training_all(self) -> None:
        """Train classifiers for all available templates."""
        clf = self.clf
        for clf_cls in CLF_TEMPLATES:
            self.clf = clf_cls()
            self.pipeline_training()
        self.clf = clf

    def _evaluate_and_save(
        self,
        x_ls: list[np.ndarray],
        y_ls: list[np.ndarray],
        index_ls: list[np.ndarray],
        name: str,
    ) -> None:
        """Evaluate model on data split and save results."""
        y_true_ls = [y[index] for y, index in zip(y_ls, index_ls, strict=False)]
        y_prob_ls = [
            self.clf.predict(x=x, index=index, batch_size=self.config.batch_size)
            for x, index in zip(x_ls, index_ls, strict=False)
        ]

        y_true = np.concatenate(y_true_ls)
        y_prob = np.concatenate(y_prob_ls)
        y_pred = (y_prob > self.config.pcutoff).astype(int)

        save_evaluation_results(
            y_true,
            y_prob,
            y_pred,
            self.config.behav_name,
            self.config.pcutoff,
            self.eval_dir,
            name,
            index_ls,
        )

    #################################################
    # Inference Pipeline
    #################################################

    def pipeline_inference(self, x_df: pd.DataFrame) -> pd.DataFrame:
        """Run inference on feature dataframe.

        Parameters
        ----------
        x_df : pd.DataFrame
            Unprocessed features dataframe.

        Returns:
        -------
        pd.DataFrame
            Predictions with probability and label columns.
        """
        index = x_df.index
        x = preproc_x_transform(x_df.to_numpy(), self.preproc_fp)

        self.clf = load(self.clf_fp)

        y_prob = self.clf.predict(
            x=x,
            index=np.arange(x.shape[0]),
            batch_size=self.config.batch_size,
        )
        y_pred = (y_prob > self.config.pcutoff).astype(int)

        pred_df = BehavPredictedDf.init_df(pd.Series(index))
        pred_df[(self.config.behav_name, PROB)] = y_prob
        pred_df[(self.config.behav_name, PRED)] = y_pred

        return pred_df
