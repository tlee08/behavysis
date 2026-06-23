"""Features DataFrame for extracted behavioral features."""

from behavysis.constants import FEATURES, FRAME

from .df_mixin import DFMixin


class FeaturesDf(DFMixin):
    """FeaturesDf."""

    is_nullable = False
    index_names = (FRAME,)
    column_names = (FEATURES,)
