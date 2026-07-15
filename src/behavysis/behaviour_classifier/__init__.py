"""Behavysis behavioural classifier — training and inference."""

from .behaviour_classifier import (
    init_classifier,
    make_eval_report_choose_model,
    predict,
    train_all_models,
    train_model,
)
from .config import ClassifierContract, TrainingRecipe
from .data import (
    agg_eval_df_by_bouts,
    load_all_data,
    load_feature_names,
    stratified_split_by_group,
)
from .evaluation import make_eval_report
from .registry import MODEL_REGISTRY
from .storage import ClassifierFp

__all__ = [
    "MODEL_REGISTRY",
    "ClassifierContract",
    "ClassifierFp",
    "TrainingRecipe",
    "agg_eval_df_by_bouts",
    "init_classifier",
    "load_all_data",
    "load_feature_names",
    "make_eval_report",
    "make_eval_report_choose_model",
    "predict",
    "stratified_split_by_group",
    "train_all_models",
    "train_model",
]
