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
    def __init__(self, root_dir: Path):
        self._root_dir = root_dir.resolve()

    # ── root ────────────────────────────────────────────────────---------

    def root_dir(self) -> Path:
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
        return self.root_dir() / TRAINING_DATA

    def features_dir(self) -> Path:
        return self.training_data_dir() / FEATURES_EXTRACTED_DIR

    def labels_dir(self) -> Path:
        return self.training_data_dir() / BEHAVIOUR_SCORED_DIR

    # ── model level ─────────────────────────────────────────────────-----
    def models_dir(self) -> Path:
        return self.root_dir() / CLASSIFIERS

    def model_dir(self, model_name: str, iteration: int) -> Path:
        """Directory for a training run: classifiers/{name}-{iteration:03d}."""
        return self.models_dir() / f"{model_name}-{iteration:03d}"

    def config_fp(self, model_name: str, iteration: int) -> Path:
        return self.model_dir(model_name, iteration) / "config.yaml"

    def eval_dir(self, model_name: str, iteration: int) -> Path:
        return self.model_dir(model_name, iteration) / "evaluation"

    # ── active model ─────────────────────────────────────────────────----

    def active_model_dir(self) -> Path:
        """Directory for a training run: classifiers/{name}-{iteration:03d}."""
        active = ClassifierActive.read_yaml(self.active_fp())
        return self.model_dir(active.model_name, active.iteration)

    def active_config_fp(self) -> Path:
        return self.active_model_dir() / "config.yaml"

    def active_eval_dir(self) -> Path:
        return self.active_model_dir() / "evaluation"
