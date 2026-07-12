"""Model registry: name → adapter factory, plus default hyperparameters.

Each entry defines a **full** pipeline builder — a callable that receives
the ``TrainingRecipe`` and returns an ``ImbPipeline``.  A shared
``_feature_selection_steps`` helper appends ``VarianceThreshold`` and
``SelectFromModel`` when the config requests them; model-type builders
can use it or define their own feature-selection logic.

Hyperparameters use native sklearn ``stepname__param`` prefixing and
are passed straight through to ``GridSearchCV``.
"""

from collections.abc import Callable

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

from .adapter import BaseAdapter, SklearnAdapter

# ── registry ─────────────────────────────────────────────────────────


MODEL_REGISTRY: dict[str, tuple[Callable[[], BaseAdapter], dict[str, list[object]]]] = {
    "rf": (
        lambda: SklearnAdapter(
            ImbPipeline(
                [
                    ("undersampler", RandomUnderSampler()),
                    ("oversampler", RandomOverSampler()),
                    ("var_filter", VarianceThreshold()),
                    (
                        "clf",
                        RandomForestClassifier(random_state=42, verbose=1, n_jobs=4),
                    ),
                ]
            )
        ),
        {
            "undersampler__sampling_strategy": [0.2],
            "oversampler__sampling_strategy": [0.4],
            "var_filter__threshold": [0.0],
            "clf__n_estimators": [200, 500],
            "clf__max_depth": [4, 8, 16],
            "clf__class_weight": ["balanced", None],
        },
    ),
    "logreg": (
        lambda: SklearnAdapter(
            ImbPipeline(
                [
                    ("undersampler", RandomUnderSampler()),
                    ("oversampler", RandomOverSampler()),
                    ("scaler", MinMaxScaler()),
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
            )
        ),
        {
            "undersampler__sampling_strategy": [0.2],
            "oversampler__sampling_strategy": [0.4],
            "var_filter__threshold": [0.0],
            "clf__C": [0.1, 1.0, 10.0],
            "clf__penalty": ["l2", None],
            "clf__max_iter": [1000],
        },
    ),
    "xgb": (
        lambda: SklearnAdapter(
            ImbPipeline(
                [
                    ("undersampler", RandomUnderSampler()),
                    ("oversampler", RandomOverSampler()),
                    ("var_filter", VarianceThreshold()),
                    ("clf", XGBClassifier(random_state=42, verbosity=1)),
                ]
            )
        ),
        {
            "undersampler__sampling_strategy": [0.2],
            "oversampler__sampling_strategy": [0.4],
            "var_filter__threshold": [0.0],
            "clf__n_estimators": [200, 500],
            "clf__max_depth": [4, 8, 16],
            "clf__learning_rate": [0.01, 0.1, 0.3],
        },
    ),
}
