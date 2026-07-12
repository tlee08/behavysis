"""Model registry: name → adapter factory, plus default hyperparameters."""

from collections.abc import Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .adapter import BaseAdapter, SklearnAdapter

ModelFactory = Callable[[], BaseAdapter]

MODEL_REGISTRY: dict[str, tuple[ModelFactory, dict[str, list[object]]]] = {
    "rf": (
        lambda: SklearnAdapter(RandomForestClassifier),
        {
            "n_estimators": [100, 200, 500, 1000],
            "max_depth": [4, 8, None],
            "class_weight": ["balanced", None],
            "random_state": [42],
            "n_jobs": [-1],
            "verbose": [1],
        },
    ),
    "logreg": (
        lambda: SklearnAdapter(LogisticRegression),
        {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l2", None],
            "max_iter": [1000],
            "random_state": [42],
            "n_jobs": [-1],
            "verbose": [1],
        },
    ),
    "xgb": (
        lambda: SklearnAdapter(XGBClassifier),
        {
            "n_estimators": [100, 200, 500, 1000],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.1, 0.3],
            "random_state": [42],
            "n_jobs": [-1],
            "verbosity": [1],
        },
    ),
}
