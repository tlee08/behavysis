"""
Utility functions.
"""

from enum import Enum

from behavysis.df_classes.df_mixin import DFMixin, FramesIN


class FeaturesCN(Enum):
    FEATURES = "features"


class FeaturesDf(DFMixin):
    """
    Mixin for features DF
    (generated from SimBA feature extraction)
    functions.
    """

    NULLABLE = False
    IN = FramesIN
    CN = FeaturesCN
