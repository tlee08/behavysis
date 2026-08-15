"""YAML-serialised Pydantic models for classifier configuration."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

from pydantic import PositiveFloat, PositiveInt  # noqa: TC002

from behavysis.models.base import YamlModel


class ClassifierContract(YamlModel):
    """Shared classifier contract (contract.yaml per classifier).

    The single source of truth for what every model in the classifier
    trains on: the behaviour, the feature set name, and the evaluation
    metric. Authored before training and never auto-modified.
    """

    behaviour_name: str
    training_project_path: Path
    feature_set: str = "generic"
    eval_metric: str = "f2"
    eval_metric_higher_better: bool = True


class ActiveModel(YamlModel):
    """Stores which model to use."""

    model_name: str


class ModelRecipe(YamlModel):
    """Human-authored model recipe (recipe.yaml).

    Model-specific hyperparameters live in ``hyperparameters``. Every value
    must be a list — even single-option entries (e.g. ``random_state: [42]``).
    All values are grid-searched via ``GridSearchCV`` at fit time.
    """

    behaviour_name: str

    model_type: str
    model_name: str

    # Train/val/test split parameters
    seed: int = 42
    test_split: float = 0.2
    val_split: float = 0.2

    # Sub-sampling parameters
    stride_frames: PositiveInt = 8
    under_sampling_strategy: PositiveFloat | None = 1.0

    # Pcutoff calibration (affects the pcutoff value used at inference time)
    target_recall: float = 0.98

    # Prediction post-processing parameters
    pcutoff: PositiveFloat = 0.2
    smoothing_frames: PositiveInt = 2
    min_gap: PositiveInt = 3
    min_bout: PositiveInt = 3
