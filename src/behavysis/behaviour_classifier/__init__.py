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
from .config import ActiveModel, ClassifierContract, ModelRecipe
from .data import (
    load_all_data,
    smooth_prob,
    stratified_split_by_group,
)
from .evaluation import make_eval_result
from .registry import MODEL_REGISTRY
from .storage import ClassifierPaths

__all__ = [
    "MODEL_REGISTRY",
    "ActiveModel",
    "ClassifierContract",
    "ClassifierPaths",
    "ModelRecipe",
    "list_models",
    "load_all_data",
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
