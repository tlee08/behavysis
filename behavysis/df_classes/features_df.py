"""
Utility functions.
"""

from enum import Enum

from behavysis.df_classes.keypoints_df import FramesIN
from behavysis.utils.df_mixin import DFMixin


class FeaturesCN(Enum):
    FEATURES = "features"


class FeaturesDf(DFMixin):
    NULLABLE = False
    IN = FramesIN
    CN = FeaturesCN
