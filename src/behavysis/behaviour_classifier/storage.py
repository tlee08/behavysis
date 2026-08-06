"""On-disk path functions for a self-contained behaviour classifier.

A classifier lives entirely inside its own directory (``clf_dir``).
Training data lives inside it, mirroring the inference pipeline's stage
folders.  Each training run produces a flat model directory::

    {clf_dir}/
        contract.yaml                  # shared behaviour + feature contract
        active.yaml                    # model to use
        models/
            rf/
                recipe.yaml            # human-authored recipe
                model.joblib           # fitted sklearn Pipeline
                evaluation/            # plots, parquet eval data
            logreg/
                ...
            xgb/
                ...
    {path/to/my/training_data}/
        5_features_extracted/
        7_behaviour_scored/
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from behavysis.constants import BEHAVIOUR_SCORED_DIR, FEATURES_EXTRACTED_DIR

from .config import ActiveModel, ClassifierContract

MODELS_DIR = "models"
TRAINING_DATA = "training_data"

if TYPE_CHECKING:
    from pathlib import Path


class ClassifierPaths:
    """Path helper for a classifier's on-disk layout."""

    _contract_fp: Path

    def __init__(self, contract_fp: Path) -> None:
        """__init__."""
        self._contract_fp = contract_fp
        self._contract = ClassifierContract.read_yaml(contract_fp)

    # -- contract contents ------------------------------------------------

    def contract(self) -> ClassifierContract:
        """Contract."""
        return self._contract

    # -- root -------------------------------------------------------------

    def root_dir(self) -> Path:
        """root_dir."""
        return self._contract_fp.parent

    # -- root level -------------------------------------------------------

    def contract_fp(self) -> Path:
        """Shared behaviour + feature contract."""
        return self._contract_fp

    def active_fp(self) -> Path:
        """Stores which model to use."""
        return self.root_dir() / "active.yaml"

    # -- training data ----------------------------------------------------

    def features_dir(self) -> Path:
        """features_dir."""
        return (
            self.root_dir()
            / self.contract().training_project_path
            / FEATURES_EXTRACTED_DIR
            / self.contract().feature_set
        )

    def labels_dir(self) -> Path:
        """labels_dir."""
        return (
            self.root_dir()
            / self.contract().training_project_path
            / BEHAVIOUR_SCORED_DIR
        )

    # -- model level ------------------------------------------------------

    def models_dir(self) -> Path:
        """models_dir."""
        return self.root_dir() / MODELS_DIR

    def model_dir(self, model_name: str) -> Path:
        """Directory for a training run: models/{name}."""
        return self.models_dir() / model_name

    def recipe_fp(self, model_name: str) -> Path:
        """recipe_fp."""
        return self.model_dir(model_name) / "recipe.yaml"

    def eval_dir(self, model_name: str) -> Path:
        """eval_dir."""
        return self.model_dir(model_name) / "evaluation"

    # -- active model -----------------------------------------------------

    def active_model_dir(self) -> Path:
        """Directory for a training run: models/{name}."""
        active = ActiveModel.read_yaml(self.active_fp())
        return self.model_dir(active.model_name)

    def active_recipe_fp(self) -> Path:
        """active_recipe_fp."""
        return self.active_model_dir() / "recipe.yaml"

    def active_eval_dir(self) -> Path:
        """active_eval_dir."""
        return self.active_model_dir() / "evaluation"
