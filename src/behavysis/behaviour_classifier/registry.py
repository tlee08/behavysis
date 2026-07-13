"""Model registry: name → adapter factory."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    HalvingRandomSearchCV,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from behavysis.utils import get_gpu_device

from .adapter import BaseAdapter, SklearnAdapter

# ── registry ─────────────────────────────────────────────────────────


MODEL_REGISTRY: dict[str, Callable[[Path], BaseAdapter]] = {
    "baseline": lambda config_fp: SklearnAdapter(
        GridSearchCV(
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
        config_fp,
    ),
    "rf": lambda config_fp: SklearnAdapter(
        HalvingRandomSearchCV(
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
            scoring="average_precision",
            cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42),
            random_state=42,
            n_jobs=1,
            verbose=3,
        ),
        config_fp,
    ),
    "logreg": lambda config_fp: SklearnAdapter(
        HalvingRandomSearchCV(
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
                            max_features=200,
                        ),
                    ),
                    ("clf", LogisticRegression(random_state=42, verbose=1)),
                ]
            ),
            {
                "var_filter__threshold": [0.0],
                "clf__C": [0.1, 1.0, 10.0],
                "clf__penalty": ["l2", None],
                "clf__max_iter": [1000],
            },
            scoring="average_precision",
            cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42),
            random_state=42,
            n_jobs=1,
            verbose=3,
        ),
        config_fp,
    ),
    "xgb": lambda config_fp: SklearnAdapter(
        HalvingRandomSearchCV(
            Pipeline(
                [
                    ("var_filter", VarianceThreshold()),
                    (
                        "clf",
                        XGBClassifier(
                            tree_method="hist",
                            eval_metric="aucpr",
                            n_jobs=4,
                            random_state=42,
                            verbosity=2,
                        ),
                    ),
                ]
            ),
            {
                "var_filter__threshold": [0.0],
                "clf__max_depth": [3, 4, 6],
                "clf__learning_rate": [0.02, 0.1],
                "clf__n_estimators": [400, 800],
                "clf__min_child_weight": [1, 10, 30, 50],
                "clf__subsample": [0.6, 0.8, 1.0],
                "clf__colsample_bytree": [0.3, 0.5, 0.7],
                "clf__gamma": [0, 0.5, 2.0],
                "clf__reg_lambda": [1.0, 3.0, 10.0],
                "clf__scale_pos_weight": [1, 10, 40],
            },
            scoring="average_precision",
            cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42),
            random_state=42,
            n_jobs=1,
            verbose=3,
        ),
        config_fp,
    ),
    "xgb_v2": lambda config_fp: SklearnAdapter(
        HalvingRandomSearchCV(
            Pipeline(
                [
                    ("var_filter", VarianceThreshold(threshold=0.0)),
                    (
                        "selector",
                        SelectFromModel(
                            XGBClassifier(
                                tree_method="hist",
                                device=get_gpu_device(),
                                n_estimators=100,
                                max_depth=4,
                                n_jobs=-1,
                                random_state=42,
                                verbosity=2,
                            ),
                            threshold=-np.inf,
                            max_features=300,
                        ),
                    ),
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
                "clf__n_estimators": [400, 800],
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
            scoring="average_precision",
            cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42),
            random_state=42,
            n_jobs=1,
            verbose=3,
        ),
        config_fp,
    ),
}

# ── registry ─────────────────────────────────────────────────────────


# Models trained by ``train_all_models``. Others remain callable manually.
ROUTINE_MODELS = ["xgb_v2"]
