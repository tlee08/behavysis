"""Collated analysis DataFrames for cross-experiment aggregation."""

from enum import Enum

from behavysis.utils.df_mixin import DFMixin


class AnalysisSummaryCollatedIN(Enum):
    """AnalysisSummaryCollatedIN."""

    EXPERIMENT = "experiment"
    INDIVIDUALS = "individuals"
    MEASURES = "measures"


class AnalysisSummaryCollatedCN(Enum):
    """AnalysisSummaryCollatedCN."""

    AGGS = "aggs"


class AnalysisBinnedCollatedIN(Enum):
    """AnalysisBinnedCollatedIN."""

    BIN_SEC = "bin_sec"


class AnalysisBinnedCollatedCN(Enum):
    """AnalysisBinnedCollatedCN."""

    EXPERIMENT = "experiment"
    INDIVIDUALS = "individuals"
    MEASURES = "measures"
    AGGS = "aggs"


class AnalysisSummaryCollatedDf(DFMixin):
    """AnalysisSummaryCollatedDf."""

    NULLABLE = False
    IN = AnalysisSummaryCollatedIN
    CN = AnalysisSummaryCollatedCN


class AnalysisBinnedCollatedDf(DFMixin):
    """AnalysisBinnedCollatedDf."""

    NULLABLE = False
    IN = AnalysisBinnedCollatedIN
    CN = AnalysisBinnedCollatedCN
