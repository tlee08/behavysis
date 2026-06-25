"""Behaviour classifier evaluation DataFrames."""

# TODO: simplify along with behav classifier

from behavysis.constants import BEHAVIOUR, FRAME, OUTCOMES

from .df_mixin import DFMixin


class BehaviourClassifierCombinedDf(DFMixin):
    """BehavClassifierCombinedDf."""

    index_names = ("video", "frame")
    column_names = (BEHAVIOUR, OUTCOMES)


class BehaviourClassifierEvalDf(DFMixin):
    """BehavClassifierEvalDf."""

    is_nullable = False
    index_names = (FRAME,)
    column_names = (BEHAVIOUR, OUTCOMES)
