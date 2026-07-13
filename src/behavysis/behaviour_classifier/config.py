"""YAML-serialised Pydantic models for classifier configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel

from behavysis.behaviour_classifier.adapter import ModelStrOptions


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

    model_name: str
    iteration: int


class TrainingRecipe(YamlModel):
    """Human-authored training recipe (config.yaml per iteration).

    Model-specific hyperparameters live in ``hyperparameters``. Every value
    must be a list — even single-option entries (e.g. ``random_state: [42]``).
    All values are grid-searched via ``GridSearchCV`` at fit time.
    """

    behaviour_name: str
    model_name: str
    model_type: ModelStrOptions
    seed: int = 42
    test_split: float = 0.2
    val_split: float = 0.2
    target_recall: float = 0.95
    pcutoff: float = 0.2
    downsample_n: int = 100_000
