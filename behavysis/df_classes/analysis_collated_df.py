from enum import Enum

from behavysis.df_classes.df_mixin import DFMixin


class AnalysisSummaryCollatedIN(Enum):
    EXPERIMENT = "experiment"
    INDIVIDUALS = "individuals"
    MEASURES = "measures"


class AnalysisSummaryCollatedCN(Enum):
    AGGS = "aggs"


class AnalysisBinnedCollatedIN(Enum):
    BIN_SEC = "bin_sec"


class AnalysisBinnedCollatedCN(Enum):
    EXPERIMENT = "experiment"
    INDIVIDUALS = "individuals"
    MEASURES = "measures"
    AGGS = "aggs"


class AnalysisSummaryCollatedDf(DFMixin):
    NULLABLE = False
    IN = AnalysisSummaryCollatedIN
    CN = AnalysisSummaryCollatedCN


class AnalysisBinnedCollatedDf(DFMixin):
    NULLABLE = False
    IN = AnalysisBinnedCollatedIN
    CN = AnalysisBinnedCollatedCN
