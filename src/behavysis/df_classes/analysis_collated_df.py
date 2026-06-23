"""Collated analysis DataFrames for cross-experiment aggregation."""

from behavysis.constants import (
    AGGS,
    BIN_SEC,
    EXPERIMENT,
    INDIVIDUALS,
    MEASURES,
)

from .df_mixin import DFMixin


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
