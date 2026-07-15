"""On-disk path functions for a self-contained behaviour classifier.

A classifier lives entirely inside its own directory (``clf_dir``).
Training data lives inside it, mirroring the inference pipeline's stage
folders.  Each training run produces a flat numbered model directory::

    {clf_dir}/
        contract.yaml                  # shared behaviour + feature contract
        active.yaml                    # model to use
        training_data/
            5_features_extracted/
            7_behaviour_scored/
        classifiers/
            rf-001/
                config.yaml            # human-authored recipe
                model.joblib           # fitted sklearn Pipeline
                evaluation/            # plots, parquet eval data
            rf-002/
                ...
            logreg-001/
                ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from behavysis.constants import BEHAVIOUR_SCORED_DIR, FEATURES_EXTRACTED_DIR

from .config import ClassifierActive

CLASSIFIERS = "classifiers"
TRAINING_DATA = "training_data"

if TYPE_CHECKING:
    from pathlib import Path


class ClassifierFp:
    """ClassifierFp."""

    def __init__(self, root_dir: Path) -> None:
        """__init__."""
        if not root_dir.exists():
            msg = f"Classifier root directory does not exist:{root_dir}"
            raise FileNotFoundError(msg)
        self._root_dir = root_dir.resolve()

    # ── root ────────────────────────────────────────────────────---------

    def root_dir(self) -> Path:
        """root_dir."""
        return self._root_dir

    # ── root level ─────────────────────────────────────────────────------

    def contract_fp(self) -> Path:
        """Shared behaviour + feature contract."""
        return self.root_dir() / "contract.yaml"

    def active_fp(self) -> Path:
        """Stores which model to use."""
        return self.root_dir() / "active.yaml"

    # ── training data ────────────────────────────────────────────────────

    def training_data_dir(self) -> Path:
        """training_data_dir."""
        return self.root_dir() / TRAINING_DATA

    def features_dir(self) -> Path:
        """features_dir."""
        return self.training_data_dir() / FEATURES_EXTRACTED_DIR

    def labels_dir(self) -> Path:
        """labels_dir."""
        return self.training_data_dir() / BEHAVIOUR_SCORED_DIR

    # ── model level ─────────────────────────────────────────────────-----
    def models_dir(self) -> Path:
        """models_dir."""
        return self.root_dir() / CLASSIFIERS

    def model_dir(self, model_name: str) -> Path:
        """Directory for a training run: classifiers/{name}."""
        return self.models_dir() / model_name

    def config_fp(self, model_name: str) -> Path:
        """config_fp."""
        return self.model_dir(model_name) / "config.yaml"

    def eval_dir(self, model_name: str) -> Path:
        """eval_dir."""
        return self.model_dir(model_name) / "evaluation"

    # ── active model ─────────────────────────────────────────────────----

    def active_model_dir(self) -> Path:
        """Directory for a training run: classifiers/{name}."""
        active = ClassifierActive.read_yaml(self.active_fp())
        return self.model_dir(active.model_name)

    def active_config_fp(self) -> Path:
        """active_config_fp."""
        return self.active_model_dir() / "config.yaml"

    def active_eval_dir(self) -> Path:
        """active_eval_dir."""
        return self.active_model_dir() / "evaluation"
