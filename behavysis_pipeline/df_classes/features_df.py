"""
Utility functions.
"""

from enum import Enum

from behavysis_pipeline.df_classes.df_mixin import DFMixin, FramesIN

####################################################################################################
# DF CONSTANTS
####################################################################################################


class FeaturesCN(Enum):
    FEATURES = "features"


####################################################################################################
# DF CLASS
####################################################################################################


class FeaturesDf(DFMixin):
    """
    Mixin for features DF
    (generated from SimBA feature extraction)
    functions.
    """

    NULLABLE = False
    IN = FramesIN
    CN = FeaturesCN
