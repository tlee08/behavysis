"""
Utility functions.
"""

from enum import Enum

from behavysis.df_classes.df_mixin import DFMixin, FramesIN


class FeaturesCN(Enum):
    FEATURES = "features"


class FeaturesDf(DFMixin):
    NULLABLE = False
    IN = FramesIN
    CN = FeaturesCN
