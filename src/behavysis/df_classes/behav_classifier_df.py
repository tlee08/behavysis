"""Behavior classifier evaluation DataFrames."""

# TODO: simplify along with behav classifier

from behavysis.df_classes.behav_df import ACTUAL, PRED, PROB, BehavPredictedDf

from .df_mixin import DFMixin


class BehavClassifierCombinedDf(DFMixin):
    """BehavClassifierCombinedDf."""

    index_names = ("video", "frame")


class BehavClassifierEvalDf(BehavPredictedDf):
    """BehavClassifierEvalDf."""

    OutcomesCols = (PROB, PRED, ACTUAL)
