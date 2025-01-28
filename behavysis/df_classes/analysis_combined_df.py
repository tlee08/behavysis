from enum import Enum

from behavysis.df_classes.df_mixin import DFMixin, FramesIN


class AnalysisCombinedCN(Enum):
    ANALYSIS = "analysis"
    INDIVIDUALS = "individuals"
    MEASURES = "measures"


class AnalysisCombinedDf(DFMixin):
    """__summary__"""

    NULLABLE = False
    IN = FramesIN
    CN = AnalysisCombinedCN
