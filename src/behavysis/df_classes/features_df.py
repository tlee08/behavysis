"""Features DataFrame for extracted behavioral features."""

from enum import Enum

from behavysis.df_classes.keypoints_df import FramesIN
from behavysis.utils.df_mixin import DFMixin


class FeaturesCN(Enum):
    """FeaturesCN."""

    FEATURES = "features"


class FeaturesDf(DFMixin):
    """FeaturesDf."""

    NULLABLE = False
    IN = FramesIN
    CN = FeaturesCN
