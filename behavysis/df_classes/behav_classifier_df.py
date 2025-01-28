"""
Utility functions.
"""

from enum import Enum

from behavysis.df_classes.df_mixin import DFMixin


class CombinedFramesIN(Enum):
    VIDEO = "video"
    FRAME = "frame"


class BehavClassifierYCN(Enum):
    BEHAVS = "behavs"


class BehavClassifierCombinedDf(DFMixin):
    IN = CombinedFramesIN


# class BehavClassifierXDf(BehavClassifierCombinedDf):
#     CN = FeaturesDf.CN


# class BehavClassifierYDf(BehavClassifierCombinedDf):
#     CN = BehavClassifierYCN
