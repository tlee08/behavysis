from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold

from behavysis.constants import EXPERIMENT, FRAME, Array1D, Array2D

_META_COLS = [EXPERIMENT, FRAME]


def select_features(
    x: Array2D,
    y: Array1D,
    variance_threshold: float,
    max_features: int,
) -> np.ndarray:
    """Return column indices to keep, fit on training data only.

    Drops low-variance columns, then optionally caps to the top
    ``max_features`` by random-forest importance. Returns all columns when
    ``feature_selection`` is disabled.
    """
    keep = np.arange(x.shape[1])

    vt = VarianceThreshold(threshold=variance_threshold)
    vt.fit(x)
    keep = keep[vt.get_support()]

    if max_features is not None and len(keep) > max_features:
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1, verbose=1
        )
        rf.fit(x[:, keep], y)
        top = np.argsort(rf.feature_importances_)[::-1][:max_features]
        keep = np.sort(keep[top])

    return keep
