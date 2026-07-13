"""Behavysis behavioural classifier — training and inference."""

from .behaviour_classifier import predict_df, train, train_all_models
from .config import ClassifierContract, TrainingRecipe
from .data import (
    agg_eval_df_by_bouts,
    load_feature_names,
    load_training_data,
    stratified_split_by_group,
)
from .evaluation import save_eval_report
from .registry import MODEL_REGISTRY

__all__ = [
    "MODEL_REGISTRY",
    "ClassifierContract",
    "TrainingRecipe",
    "agg_eval_df_by_bouts",
    "load_feature_names",
    "load_training_data",
    "predict_df",
    "save_eval_report",
    "stratified_split_by_group",
    "train",
    "train_all_models",
]
