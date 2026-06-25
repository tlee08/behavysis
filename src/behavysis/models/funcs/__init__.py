"""Process Config Models."""

from .analyse import AnalyseConfig
from .calculate_params import CalculateParamsConfig
from .classify_behavs import ClassifyBehavConfig
from .evaluate_vid import EvaluateVidConfig
from .extract_features import ExtractFeaturesConfig
from .format_vid import FormatVidConfig, VidMetadata
from .preprocess import PreprocessConfig
from .run_dlc import RunDlcConfig

__all__ = [
    "AnalyseConfig",
    "CalculateParamsConfig",
    "ClassifyBehavConfig",
    "EvaluateVidConfig",
    "ExtractFeaturesConfig",
    "FormatVidConfig",
    "PreprocessConfig",
    "RunDlcConfig",
    "VidMetadata",
]
