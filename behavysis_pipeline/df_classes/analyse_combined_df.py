from enum import Enum

from behavysis_pipeline.df_classes.df_mixin import DFMixin, FramesIN


class AnalyseCombinedCN(Enum):
    ANALYSIS = "analysis"
    INDIVIDUALS = "individuals"
    MEASURES = "measures"


class AnalyseCombinedDf(DFMixin):
    """__summary__"""

    NULLABLE = False
    IN = FramesIN
    CN = AnalyseCombinedCN
