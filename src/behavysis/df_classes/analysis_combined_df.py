"""Combined analysis DataFrame from multiple analysis types."""

from enum import Enum

from behavysis.df_classes.keypoints_df import FramesIN
from behavysis.utils.df_mixin import DFMixin


class AnalysisCombinedCN(Enum):
    ANALYSIS = "analysis"
    INDIVIDUALS = "individuals"
    MEASURES = "measures"


class AnalysisCombinedDf(DFMixin):
    NULLABLE = False
    IN = FramesIN
    CN = AnalysisCombinedCN
