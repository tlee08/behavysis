"""YAML-serialised Pydantic models for classifier configuration and metadata."""

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


class TrainingRecipe(YamlModel):
    """Human-authored training recipe (config.yaml per model_type).

    Declares the hyperparameters for a single model_type. The behaviour and
    feature contract live in the shared ``contract.yaml`` (``ClassifierContract``).
    Authored before training and never auto-modified.

    Model-specific hyperparameters live in ``hyperparameters``. Every value
    must be a list — even single-option entries (e.g. ``random_state: [42]``).
    All values are grid-searched via ``GridSearchCV`` at fit time.
    """

    model_type: str
    oversample_ratio: float = 0.2
    undersample_ratio: float = 0.4
    split_seed: int = 42
    test_split: float = 0.2
    val_split: float = 0.2
    batch_size: int = 256
    epochs: int = 100
    pcutoff: float = 0.2
    feature_selection: bool = True
    variance_threshold: float = 0.0
    max_features: int | None = None
    hyperparameters: dict[str, list[object]] = {}


class DataSummary(BaseModel):
    """Dataset shape and class balance for a trained version."""

    n_samples: int
    n_features: int
    n_features_selected: int
    n_train: int
    n_test: int
    train_pos_ratio: float
    test_pos_ratio: float


class TrainingSummary(BaseModel):
    """Training run summary for a trained version."""

    duration_seconds: float | None = None


class EvalSummary(BaseModel):
    """Evaluation metrics per split for a trained version."""

    train_accuracy: float | None = None
    train_f1_behav: float | None = None
    val_accuracy: float | None = None
    val_f1_behav: float | None = None
    test_accuracy: float | None = None
    test_f1_behav: float | None = None


class VersionMetadata(YamlModel):
    """Machine-written metadata after training (metadata.yaml per version)."""

    version: str
    framework: str  # "sklearn" or "torch"
    model_type: str
    created_at: str
    recipe: TrainingRecipe
    data: DataSummary
    training: TrainingSummary = TrainingSummary()
    evaluation: EvalSummary = EvalSummary()


class DatasetManifest(YamlModel):
    """Snapshot of what a version trained on (dataset_manifest.yaml)."""

    version: str
    dataset_hash: str | None = None
    train_ids: list[str] = []
    test_ids: list[str] = []
    n_train: int = 0
    n_test: int = 0


class ActivePointer(YamlModel):
    """Pointer to trusted version within a model_type (active.yaml)."""

    version: str
    promoted_at: str


class LeaderboardEntry(BaseModel):
    """A single model_type's ranking in the leaderboard."""

    model_type: str
    version: str
    test_f1_behav: float | None = None
    test_accuracy: float | None = None
    train_f1_behav: float | None = None
    overfit_ratio: float | None = None


class Leaderboard(YamlModel):
    """Cross-model_type comparison (leaderboard.yaml per behaviour)."""

    behaviour_name: str
    generated_at: str
    rankings: list[LeaderboardEntry] = []


class ProductionPointer(YamlModel):
    """Deployed model pointer (production.yaml per classifier).

    Records which ``model_type`` is deployed. Written only by
    ``promote_to_production`` — not hand-edited. The version is resolved from
    that model_type's ``active.yaml``; the behaviour and feature contract are
    resolved from the classifier's ``contract.yaml``.
    """

    model_type: str
    promoted_at: str
