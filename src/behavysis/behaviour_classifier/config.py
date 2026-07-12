"""YAML-serialised Pydantic models for classifier configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import yaml
from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path


class YamlModel(BaseModel):
    """Base model with YAML read/write helpers."""

    @classmethod
    def read_yaml(cls, fp: Path) -> Self:
        """Read the model from a YAML file."""
        return cls.model_validate(yaml.safe_load(fp.read_text()))

    def write_yaml(self, fp: Path) -> None:
        """Write the model to a YAML file."""
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(yaml.dump(self.model_dump(), default_flow_style=False))


class ClassifierContract(YamlModel):
    """Shared classifier contract (contract.yaml per classifier).

    The single source of truth for what every model_type in the classifier
    trains on: the behaviour and the feature contract (``individuals`` /
    ``bodyparts``). Authored before training and never auto-modified.
    """

    behaviour_name: str
    individuals: list[str]
    bodyparts: list[str]


class ClassifierActive(YamlModel):
    """Stores which model to use."""

    name: str
    iteration: int


class TrainingRecipe(YamlModel):
    """Human-authored training recipe (config.yaml per iteration).

    Model-specific hyperparameters live in ``hyperparameters``. Every value
    must be a list — even single-option entries (e.g. ``random_state: [42]``).
    All values are grid-searched via ``GridSearchCV`` at fit time.
    """

    name: str  # classifier name, e.g. "rf"
    seed: int = 42
    test_split: float = 0.2
    val_cv_folds: int = 3
    pcutoff: float = 0.2
    feature_selection: bool = True  # only used by TorchAdapter
    variance_threshold: float = 0.0  # only used by TorchAdapter
    max_features: int | None = None  # only used by TorchAdapter
    batch_size: int = 256  # only used by TorchAdapter
    epochs: int = 100  # only used by TorchAdapter
    val_split: float = 0.2  # only used by TorchAdapter
    hyperparameters: dict[str, list[object]] = {}
