"""Process Config Models."""

from .analyse import AnalyseConfig
from .calculate_params import CalculateParamsConfig
from .classify_behaviour import ClassifyBehaviourConfig
from .evaluate_vid import EvaluateVidConfig
from .extract_features import ExtractFeaturesConfig
from .format_vid import FormatVidConfig, VidMetadata
from .preprocess import PreprocessConfig
from .run_dlc import RunDlcConfig

__all__ = [
    "AnalyseConfig",
    "CalculateParamsConfig",
    "ClassifyBehaviourConfig",
    "EvaluateVidConfig",
    "ExtractFeaturesConfig",
    "FormatVidConfig",
    "PreprocessConfig",
    "RunDlcConfig",
    "VidMetadata",
]
