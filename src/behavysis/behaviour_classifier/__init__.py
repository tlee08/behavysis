"""Behavysis behavioural classifier — training, versioning, promotion, inference."""

from .behaviour_classifier import (
    BehaviourClassifier,
    promote,
    promote_to_best,
    promote_to_production,
    regenerate_leaderboard,
    train,
    train_all_models,
)
from .config import (
    ActivePointer,
    ClassifierContract,
    DatasetManifest,
    Leaderboard,
    LeaderboardEntry,
    ProductionPointer,
    TrainingRecipe,
    VersionMetadata,
)
from .data import load_feature_names, load_training_data, stratified_split_by_video
from .registry import MODEL_REGISTRY

__all__ = [
    "MODEL_REGISTRY",
    "ActivePointer",
    "BehaviourClassifier",
    "ClassifierContract",
    "DatasetManifest",
    "Leaderboard",
    "LeaderboardEntry",
    "ProductionPointer",
    "TrainingRecipe",
    "VersionMetadata",
    "load_feature_names",
    "load_training_data",
    "promote",
    "promote_to_best",
    "promote_to_production",
    "regenerate_leaderboard",
    "stratified_split_by_video",
    "train",
    "train_all_models",
]
