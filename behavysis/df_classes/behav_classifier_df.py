"""
Utility functions.
"""

from enum import Enum

from behavysis.df_classes.behav_df import BehavPredictedDf
from behavysis.utils.df_mixin import DFMixin


class CombinedFramesIN(Enum):
    VIDEO = "video"
    FRAME = "frame"


class BehavClassifierYCN(Enum):
    BEHAVS = "behavs"


class OutcomesEvalCols(Enum):
    PROB = "prob"
    PRED = "pred"
    ACTUAL = "actual"


class BehavClassifierCombinedDf(DFMixin):
    IN = CombinedFramesIN


class BehavClassifierEvalDf(BehavPredictedDf):
    OutcomesCols = OutcomesEvalCols
