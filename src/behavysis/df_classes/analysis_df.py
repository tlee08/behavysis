"""Frame-by-frame analysis DataFrame."""

from behavysis.constants import FRAME, INDIVIDUALS, MEASURES

from .df_mixin import DFMixin


class AnalysisDf(DFMixin):
    """AnalysisDf."""

    is_nullable = False
    index_names = (FRAME,)
    column_names = (INDIVIDUALS, MEASURES)
