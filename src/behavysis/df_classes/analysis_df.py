"""Frame-by-frame analysis DataFrame."""

from behavysis.constants import (
    AGGS,
    ANALYSIS,
    BIN_SEC,
    EXPERIMENT,
    FRAME,
    INDIVIDUALS,
    MEASURES,
)

from .df_mixin import DFMixin


class AnalysisDf(DFMixin):
    """AnalysisDf."""

    is_nullable = False
    index_names = (FRAME,)
    column_names = (INDIVIDUALS, MEASURES)


class AnalysisCombinedDf(DFMixin):
    """AnalysisCombinedDf."""

    is_nullable = False
    index_names = (FRAME,)
    column_names = (ANALYSIS, INDIVIDUALS, MEASURES)


class AnalysisSummaryCollatedDf(DFMixin):
    """AnalysisSummaryCollatedDf."""

    is_nullable = False
    index_names = (EXPERIMENT, INDIVIDUALS, MEASURES)
    column_names = (AGGS,)


class AnalysisBinnedCollatedDf(DFMixin):
    """AnalysisBinnedCollatedDf."""

    is_nullable = False
    index_names = (BIN_SEC,)
    column_names = (EXPERIMENT, INDIVIDUALS, MEASURES, AGGS)
