"""Behavior classifier evaluation DataFrames."""

from enum import Enum

from behavysis.df_classes.behav_df import BehavPredictedDf
from behavysis.utils.df_mixin import DFMixin


class CombinedFramesIN(Enum):
    """CombinedFramesIN."""

    VIDEO = "video"
    FRAME = "frame"


class BehavClassifierYCN(Enum):
    """BehavClassifierYCN."""

    BEHAVS = "behavs"


class OutcomesEvalCols(Enum):
    """OutcomesEvalCols."""

    PROB = "prob"
    PRED = "pred"
    ACTUAL = "actual"


class BehavClassifierCombinedDf(DFMixin):
    """BehavClassifierCombinedDf."""

    IN = CombinedFramesIN


class BehavClassifierEvalDf(BehavPredictedDf):
    """BehavClassifierEvalDf."""

    OutcomesCols = OutcomesEvalCols
