"""YAML-serialised Pydantic models for classifier configuration and metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path


class TrainingRecipe(BaseModel):
    """Human-authored training recipe (config.yaml per model_type).

    Declares the behaviour, the feature contract (``individuals``/``bodyparts``
    the model is trained on) and hyperparameters. Authored before training and
    never auto-modified. ``behaviour_name``, ``individuals`` and ``bodyparts``
    are required — there are no implied defaults.
    """

    model_type: str
    behaviour_name: str
    individuals: list[str]
    bodyparts: list[str]
    seed: int = 42
    oversample_ratio: float = 0.2
    undersample_ratio: float = 0.4
    test_split: float = 0.2
    val_split: float = 0.2
    batch_size: int = 256
    epochs: int = 100
    pcutoff: float = 0.2

    @classmethod
    def read_yaml(cls, fp: Path) -> TrainingRecipe:
        return cls.model_validate(yaml.safe_load(fp.read_text()))

    def write_yaml(self, fp: Path) -> None:
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(yaml.dump(self.model_dump(), default_flow_style=False))


class ResolvedHyperparams(BaseModel):
    seed: int
    batch_size: int
    epochs: int
    oversample_ratio: float
    undersample_ratio: float
    test_split: float
    val_split: float


class DataSummary(BaseModel):
    n_samples: int
    n_features: int
    n_train: int
    n_val: int
    n_test: int
    train_pos_ratio: float
    test_pos_ratio: float


class TrainingSummary(BaseModel):
    duration_seconds: float | None = None


class EvalSummary(BaseModel):
    train_accuracy: float | None = None
    train_f1_behav: float | None = None
    val_accuracy: float | None = None
    val_f1_behav: float | None = None
    test_accuracy: float | None = None
    test_f1_behav: float | None = None


class VersionMetadata(BaseModel):
    """Machine-written metadata after training (metadata.yaml per version)."""

    version: str
    framework: str  # "sklearn" or "torch"
    model_type: str
    created_at: str
    resolved: ResolvedHyperparams
    data: DataSummary
    training: TrainingSummary = TrainingSummary()
    evaluation: EvalSummary = EvalSummary()

    @classmethod
    def read_yaml(cls, fp: Path) -> VersionMetadata:
        return cls.model_validate(yaml.safe_load(fp.read_text()))

    def write_yaml(self, fp: Path) -> None:
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(yaml.dump(self.model_dump(), default_flow_style=False))


class DatasetManifest(BaseModel):
    """Snapshot of what a version trained on (dataset_manifest.yaml)."""

    version: str
    dataset_hash: str | None = None
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0

    @classmethod
    def read_yaml(cls, fp: Path) -> DatasetManifest:
        return cls.model_validate(yaml.safe_load(fp.read_text()))

    def write_yaml(self, fp: Path) -> None:
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(yaml.dump(self.model_dump(), default_flow_style=False))


class ActivePointer(BaseModel):
    """Pointer to trusted version within a model_type (active.yaml)."""

    version: str
    promoted_at: str

    @classmethod
    def read_yaml(cls, fp: Path) -> ActivePointer:
        return cls.model_validate(yaml.safe_load(fp.read_text()))

    def write_yaml(self, fp: Path) -> None:
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(yaml.dump(self.model_dump(), default_flow_style=False))


class LeaderboardEntry(BaseModel):
    model_type: str
    version: str
    test_f1_behav: float | None = None
    test_accuracy: float | None = None
    train_f1_behav: float | None = None
    overfit_ratio: float | None = None


class Leaderboard(BaseModel):
    """Cross-model_type comparison (leaderboard.yaml per behaviour)."""

    behaviour_name: str
    generated_at: str
    rankings: list[LeaderboardEntry] = []

    @classmethod
    def read_yaml(cls, fp: Path) -> Leaderboard:
        return cls.model_validate(yaml.safe_load(fp.read_text()))

    def write_yaml(self, fp: Path) -> None:
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(yaml.dump(self.model_dump(), default_flow_style=False))


class ProductionPointer(BaseModel):
    """Deployed model pointer and public contract (production.yaml per classifier).

    Records the deployed model's identity (``behaviour_name``), which artifact
    to use (``model_type``/``version``), and the feature contract it was trained
    on (``individuals``/``bodyparts``) so callers can validate their extracted
    features match before classifying.
    """

    behaviour_name: str
    model_type: str
    version: str
    individuals: list[str] = []
    bodyparts: list[str] = []
    promoted_at: str

    @classmethod
    def read_yaml(cls, fp: Path) -> ProductionPointer:
        return cls.model_validate(yaml.safe_load(fp.read_text()))

    def write_yaml(self, fp: Path) -> None:
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(yaml.dump(self.model_dump(), default_flow_style=False))
