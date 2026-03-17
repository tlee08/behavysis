from enum import Enum

from behavysis.df_classes.keypoints_df import FramesIN
from behavysis.utils.df_mixin import DFMixin

FBF = "fbf"


class AnalysisCN(Enum):
    INDIVIDUALS = "individuals"
    MEASURES = "measures"


class AnalysisDf(DFMixin):
    """__summary__"""

    NULLABLE = False
    IN = FramesIN
    CN = AnalysisCN
