"""Behavior classifier evaluation DataFrames."""

# TODO: simplify along with behav classifier

from behavysis.constants import BEHAVS, FRAME, OUTCOMES

from .df_mixin import DFMixin


class BehavClassifierCombinedDf(DFMixin):
    """BehavClassifierCombinedDf."""

    index_names = ("video", "frame")
    column_names = (BEHAVS, OUTCOMES)


class BehavClassifierEvalDf(DFMixin):
    """BehavClassifierEvalDf."""

    is_nullable = False
    index_names = (FRAME,)
    column_names = (BEHAVS, OUTCOMES)
