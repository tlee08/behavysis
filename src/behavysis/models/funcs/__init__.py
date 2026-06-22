"""Process Config Models."""

from .analyse import AnalyseConfigs
from .calculate_params import CalculateParamsConfigs
from .classify_behavs import ClassifyBehavConfigs
from .evaluate_vid import EvaluateVidConfigs
from .extract_features import ExtractFeaturesConfigs
from .format_vid import FormatVidConfigs, VidMetadata
from .preprocess import PreprocessConfigs
from .run_dlc import RunDlcConfigs

__all__ = [
    "AnalyseConfigs",
    "CalculateParamsConfigs",
    "ClassifyBehavConfigs",
    "EvaluateVidConfigs",
    "ExtractFeaturesConfigs",
    "FormatVidConfigs",
    "PreprocessConfigs",
    "RunDlcConfigs",
    "VidMetadata",
]
