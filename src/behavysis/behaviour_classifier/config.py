"""YAML-serialised Pydantic models for classifier configuration."""

from __future__ import annotations

from behavysis.models.base import YamlModel


class ClassifierContract(YamlModel):
    """Shared classifier contract (contract.yaml per classifier).

    The single source of truth for what every model_type in the classifier
    trains on: the behaviour and the feature contract (``individuals`` /
    ``bodyparts``). Authored before training and never auto-modified.
    """

    behaviour_name: str
    individuals: list[str]
    bodyparts: list[str]

    eval_metric: str = "f2"
    eval_metric_higher_better: bool = True


class ClassifierActive(YamlModel):
    """Stores which model to use."""

    model_name: str


class TrainingRecipe(YamlModel):
    """Human-authored training recipe (config.yaml).

    Model-specific hyperparameters live in ``hyperparameters``. Every value
    must be a list — even single-option entries (e.g. ``random_state: [42]``).
    All values are grid-searched via ``GridSearchCV`` at fit time.
    """

    behaviour_name: str

    model_type: str

    model_name: str

    seed: int = 42
    test_split: float = 0.2
    val_split: float = 0.2
    downsample_n: int = 100_000

    target_recall: float = 0.98
    pcutoff: float = 0.2
    smoothing_frames: int = 1
