"""YAML-serialised Pydantic models for classifier configuration."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Literal

from pydantic import (  # noqa: TC002
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

from behavysis.models.base import YamlModel

FRAME_AWARE = "frame-aware"
HYSTERESIS = "hysteresis"


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


class FrameAwarePostprocessing(YamlModel):
    """Frame-level post-processing: smooth, threshold, then tidy bouts.

    Applies, in order: median smoothing (``smooth_prob``), a probability
    threshold (``prob > pcutoff``), then gap-filling and short-bout dropping
    (``smooth_pred_bout``).
    """

    pcutoff: PositiveFloat = 0.1
    smoothing_frames: NonNegativeInt = 2
    min_gap_frames: NonNegativeInt = 3
    min_bout_frames: NonNegativeInt = 3


class HysteresisPostprocessing(YamlModel):
    """Hysteresis post-processing: two thresholds, then drop short bouts.

    A frame is positive if ``prob > pcutoff`` (strong) or if ``prob >
    low_threshold`` (weak) and connected to a strong frame through weak frames.
    Then positive runs <= ``min_bout_frames`` are dropped.

    Merges brief dips (fragmentation) while keeping true separations, where
    ``prob`` falls below ``low_threshold``, as distinct bouts.
    """

    pcutoff: PositiveFloat = 0.1
    low_threshold: NonNegativeFloat = 0.01
    min_bout_frames: NonNegativeInt = 3


class ModelRecipe(YamlModel):
    """Human-authored model recipe (recipe.yaml)."""

    behaviour_name: str

    model_type: str
    model_name: str

    # Train/val/test split parameters
    seed: int = 42
    test_split: float = 0.2
    val_split: float = 0.2

    # Sub-sampling parameters
    stride_frames: PositiveInt = 2
    under_sampling_strategy: PositiveFloat | None = 1.0

    # Post-processing calibration (affects the parameters used at inference time)
    calibrate_params: bool = True
    target_recall: float = 0.95
    postprocessing_step: Literal[FRAME_AWARE, HYSTERESIS] = FRAME_AWARE  # ty: ignore[invalid-type-form]

    # Prediction post-processing parameters (auto-set)
    frame_aware: FrameAwarePostprocessing = FrameAwarePostprocessing()
    hysteresis: HysteresisPostprocessing = HysteresisPostprocessing()
