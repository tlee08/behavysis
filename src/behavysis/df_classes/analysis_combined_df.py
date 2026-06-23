"""Combined analysis DataFrame from multiple analysis types."""

from behavysis.constants import ANALYSIS, FRAME, INDIVIDUALS, MEASURES

from .df_mixin import DFMixin


class AnalysisCombinedDf(DFMixin):
    """AnalysisCombinedDf."""

    is_nullable = False
    index_names = (FRAME,)
    column_names = (ANALYSIS, INDIVIDUALS, MEASURES)
