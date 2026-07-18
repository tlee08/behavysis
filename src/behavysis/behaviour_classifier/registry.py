"""Model registry: name → adapter factory."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from behavysis.utils import get_gpu_device

from .adapter import BaseAdapter, SklearnAdapter, TabpfnAdapter

# ── registry ─────────────────────────────────────────────────────────


MODEL_REGISTRY: dict[str, Callable[[Path], BaseAdapter]] = {
    "baseline": lambda config_fp: SklearnAdapter(
        config_fp,
        search=GridSearchCV(
            Pipeline(
                [
                    ("clf", DecisionTreeClassifier(random_state=42)),
                ],
            ),
            {
                "clf__max_depth": [1],
            },
            cv=2,
            n_jobs=1,
            verbose=3,
        ),
    ),
    "rf": lambda config_fp: SklearnAdapter(
        config_fp,
        search=RandomizedSearchCV(
            Pipeline(
                [
                    ("var_filter", VarianceThreshold()),
                    (
                        "clf",
                        RandomForestClassifier(random_state=42, verbose=2, n_jobs=4),
                    ),
                ]
            ),
            {
                "var_filter__threshold": [0.0],
                "clf__n_estimators": [200, 500],
                "clf__max_depth": [4, 8, 16],
                "clf__class_weight": ["balanced", None],
            },
            n_iter=5,
            scoring="average_precision",
            cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42),
            random_state=42,
            n_jobs=1,
            verbose=3,
        ),
    ),
    "logreg": lambda config_fp: SklearnAdapter(
        config_fp,
        search=RandomizedSearchCV(
            ImbPipeline(
                [
                    ("undersampler", RandomUnderSampler(sampling_strategy=0.2)),
                    ("oversampler", RandomOverSampler(sampling_strategy=0.4)),
                    ("var_filter", VarianceThreshold()),
                    (
                        "selector",
                        SelectFromModel(
                            RandomForestClassifier(
                                n_estimators=200,
                                max_depth=8,
                                random_state=42,
                                n_jobs=4,
                                verbose=1,
                            ),
                            threshold=-np.inf,
                        ),
                    ),
                    ("clf", LogisticRegression(random_state=42, verbose=1)),
                ]
            ),
            {
                "selector__max_features": [20, 50, 100, 200],
                "var_filter__threshold": [0.0],
                "clf__C": [0.1, 1.0, 10.0],
                "clf__penalty": ["l2", None],
                "clf__max_iter": [1000],
            },
            n_iter=5,
            scoring="average_precision",
            cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42),
            random_state=42,
            n_jobs=1,
            verbose=3,
        ),
    ),
    "xgb": lambda config_fp: SklearnAdapter(
        config_fp,
        search=RandomizedSearchCV(
            Pipeline(
                [
                    ("var_filter", VarianceThreshold(threshold=0.0)),
                    (
                        "clf",
                        XGBClassifier(
                            tree_method="hist",
                            device=get_gpu_device(),
                            eval_metric="aucpr",
                            n_jobs=-1,
                            random_state=42,
                            verbosity=2,
                        ),
                    ),
                ]
            ),
            {
                "selector__estimator__importance_type": [
                    "weight",
                    "gain",
                    "total_gain",
                    "cover",
                    "total_cover",
                ],
                "selector__max_features": [300, None],
                "clf__n_estimators": [400, 800, 1200],
                "clf__learning_rate": [0.02, 0.1],
                "clf__max_depth": [3, 4, 6],
                "clf__min_child_weight": [1, 10, 30, 50],
                "clf__subsample": [0.6, 0.8, 1.0],
                "clf__colsample_bytree": [0.3, 0.5, 0.7],
                "clf__gamma": [0, 0.5, 2.0],
                "clf__reg_lambda": [1.0, 3.0, 10.0],
                "clf__scale_pos_weight": [5, 10, 20, 40, 60],
                "clf__max_delta_step": [0, 1, 3, 10],
            },
            n_iter=400,
            scoring="average_precision",
            cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42),
            random_state=42,
            n_jobs=1,
            verbose=3,
        ),
    ),
    "xgb_dart": lambda config_fp: SklearnAdapter(
        config_fp,
        search=RandomizedSearchCV(
            Pipeline(
                [
                    ("var_filter", VarianceThreshold(threshold=0.0)),
                    (
                        "clf",
                        XGBClassifier(
                            booster="dart",
                            tree_method="hist",
                            device=get_gpu_device(),
                            eval_metric="aucpr",
                            n_jobs=-1,
                            random_state=42,
                            verbosity=2,
                        ),
                    ),
                ]
            ),
            {
                "selector__max_features": [300, None],
                "clf__n_estimators": [400, 800, 1200],
                "clf__learning_rate": [0.02, 0.1],
                "clf__max_depth": [3, 4, 6],
                "clf__min_child_weight": [1, 10, 30],
                "clf__subsample": [0.6, 0.8, 1.0],
                "clf__colsample_bytree": [0.3, 0.5, 0.7],
                "clf__gamma": [0, 0.5, 2.0],
                "clf__reg_lambda": [1.0, 3.0, 10.0],
                "clf__scale_pos_weight": [5, 10, 20, 40],
                "clf__rate_drop": [0.05, 0.1, 0.2],
                "clf__skip_drop": [0.3, 0.5, 0.7],
                "clf__max_delta_step": [0, 1, 3, 10],
            },
            n_iter=200,
            scoring="average_precision",
            cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42),
            random_state=42,
            n_jobs=1,
            verbose=3,
        ),
    ),
    "xgb_smote": lambda config_fp: SklearnAdapter(
        config_fp,
        search=RandomizedSearchCV(
            ImbPipeline(
                [
                    ("var_filter", VarianceThreshold(threshold=0.0)),
                    ("smote", SMOTE(sampling_strategy="auto", random_state=42)),
                    (
                        "clf",
                        XGBClassifier(
                            tree_method="hist",
                            device=get_gpu_device(),
                            eval_metric="aucpr",
                            grow_policy="lossguide",
                            n_jobs=-1,
                            random_state=42,
                            verbosity=2,
                        ),
                    ),
                ]
            ),
            {
                "smote__k_neighbors": [3, 5, 7],
                "clf__n_estimators": [400, 800, 1200],
                "clf__learning_rate": [0.02, 0.1],
                "clf__max_depth": [0, 4, 6],
                "clf__max_leaves": [31, 63, 127],
                "clf__min_child_weight": [1, 10, 30],
                "clf__subsample": [0.6, 0.8, 1.0],
                "clf__colsample_bytree": [0.3, 0.5, 0.7],
                "clf__gamma": [0, 0.5, 2.0],
                "clf__reg_lambda": [1.0, 3.0, 10.0],
                "clf__scale_pos_weight": [1, 3, 7],
                "clf__max_delta_step": [0, 1, 3, 10],
            },
            n_iter=80,
            scoring="average_precision",
            cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42),
            random_state=42,
            n_jobs=1,
            verbose=3,
        ),
    ),
    "xgb_calibrated": lambda config_fp: SklearnAdapter(
        config_fp,
        search=RandomizedSearchCV(
            Pipeline(
                [
                    ("var_filter", VarianceThreshold(threshold=0.0)),
                    (
                        "clf",
                        CalibratedClassifierCV(
                            estimator=XGBClassifier(
                                tree_method="hist",
                                device=get_gpu_device(),
                                eval_metric="aucpr",
                                n_jobs=-1,
                                random_state=42,
                                verbosity=2,
                            ),
                            cv=3,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "clf__method": ["isotonic", "sigmoid"],
                "clf__estimator__n_estimators": [400, 800],
                "clf__estimator__learning_rate": [0.02, 0.1],
                "clf__estimator__max_depth": [3, 4, 6],
                "clf__estimator__min_child_weight": [1, 10, 30],
                "clf__estimator__subsample": [0.6, 0.8, 1.0],
                "clf__estimator__colsample_bytree": [0.3, 0.5, 0.7],
                "clf__estimator__reg_lambda": [1.0, 3.0, 10.0],
                "clf__estimator__scale_pos_weight": [5, 10, 20, 40],
            },
            n_iter=40,
            scoring="average_precision",
            cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42),
            random_state=42,
            n_jobs=1,
            verbose=3,
        ),
    ),
    "tabpfn": lambda config_fp: TabpfnAdapter(
        config_fp=config_fp,
        n_estimators=8,
        balance_probabilities=True,
        device=get_gpu_device(),
        ignore_pretraining_limits=True,
        fit_mode="fit_with_cache",
        random_state=42,
        inference_config={
            "SUBSAMPLE_SAMPLES": 100_000,
        },
    ),
    "tabpfn-large": lambda config_fp: TabpfnAdapter(
        config_fp=config_fp,
        n_estimators=16,
        balance_probabilities=True,
        device=get_gpu_device(),
        ignore_pretraining_limits=True,
        fit_mode="fit_with_cache",
        random_state=42,
        inference_config={
            "SUBSAMPLE_SAMPLES": 100_000,
        },
    ),
}

# ── registry ─────────────────────────────────────────────────────────
