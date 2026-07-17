"""Behavysis behavioural classifier — training and inference."""

from .behaviour_classifier import (
    list_models,
    make_eval_result_choose_model,
    predict,
    promote_best,
    train_all_models,
    train_model,
    write_contract,
)
from .config import ClassifierContract, TrainingRecipe
from .data import (
    load_all_data,
    load_feature_names,
    smooth_prob,
    stratified_split_by_group,
)
from .evaluation import make_eval_result
from .registry import MODEL_REGISTRY
from .storage import ClassifierFp

__all__ = [
    "MODEL_REGISTRY",
    "ClassifierContract",
    "ClassifierFp",
    "TrainingRecipe",
    "list_models",
    "load_all_data",
    "load_feature_names",
    "make_eval_result",
    "make_eval_result_choose_model",
    "predict",
    "promote_best",
    "smooth_prob",
    "stratified_split_by_group",
    "train_all_models",
    "train_model",
    "write_contract",
]
