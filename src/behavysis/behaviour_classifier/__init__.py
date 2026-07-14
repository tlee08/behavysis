"""Behavysis behavioural classifier — training and inference."""

from .behaviour_classifier import (
    init_classifier,
    initial_train,
    make_eval_report_choose_model,
    predict_df,
    train,
    train_all_models,
)
from .config import ClassifierContract, TrainingRecipe
from .data import (
    agg_eval_df_by_bouts,
    load_feature_names,
    load_training_data,
    stratified_split_by_group,
)
from .evaluation import make_eval_report
from .registry import MODEL_REGISTRY

__all__ = [
    "MODEL_REGISTRY",
    "ClassifierContract",
    "TrainingRecipe",
    "agg_eval_df_by_bouts",
    "init_classifier",
    "initial_train",
    "load_feature_names",
    "load_training_data",
    "make_eval_report",
    "make_eval_report_choose_model",
    "predict_df",
    "stratified_split_by_group",
    "train",
    "train_all_models",
]
