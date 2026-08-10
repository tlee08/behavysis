"""Pipeline stage functions."""

from .analyse import AnalyseFunc, analyse_behaviour, distance, in_roi, social_distance
from .calculate_parameters import (
    CalculateParametersFunc,
    dur_frames_from_likelihood,
    px_per_mm,
    start_frame_from_csv,
    start_frame_from_likelihood,
    stop_frame_from_dur,
    stop_frame_from_likelihood,
)
from .classify_behaviour import classify_behaviour
from .combine_analysis import combine_analysis
from .extract_features import (
    ExtractFeaturesFunc,
    extract_generic,
    extract_hpw,
    extract_rearing,
)
from .format_video import format_video, get_video_metadata
from .preprocess import (
    PreprocessFunc,
    interpolate,
    interpolate_stationary,
    start_stop_trim,
)
from .run_dlc import run_dlc_ma

__all__ = [
    "AnalyseFunc",
    "CalculateParametersFunc",
    "ExtractFeaturesFunc",
    "PreprocessFunc",
    "analyse_behaviour",
    "classify_behaviour",
    "combine_analysis",
    "distance",
    "dur_frames_from_likelihood",
    "extract_generic",
    "extract_hpw",
    "extract_rearing",
    "format_video",
    "get_video_metadata",
    "in_roi",
    "interpolate",
    "interpolate_stationary",
    "px_per_mm",
    "run_dlc_ma",
    "social_distance",
    "start_frame_from_csv",
    "start_frame_from_likelihood",
    "start_stop_trim",
    "stop_frame_from_dur",
    "stop_frame_from_likelihood",
]
