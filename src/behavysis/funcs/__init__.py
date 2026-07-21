"""Pipeline stage functions."""

from .analyse import AnalyseFunc, distance, in_roi, social_distance
from .behaviour import analyse_behaviour, classify_single
from .calculate_parameters import (
    CalculateParametersFunc,
    dur_frames_from_likelihood,
    px_per_mm,
    start_frame_from_csv,
    start_frame_from_likelihood,
    stop_frame_from_dur,
    stop_frame_from_likelihood,
)
from .combine_analysis import combine_analysis
from .extract_features import ExtractFeaturesFunc, extract_features
from .format_video import format_video, get_video_metadata
from .preprocess import (
    PreprocessFunc,
    interpolate,
    interpolate_stationary,
    start_stop_trim,
)
from .run_dlc import ma_dlc_run_batch, ma_dlc_run_single

__all__ = [
    "AnalyseFunc",
    "CalculateParametersFunc",
    "ExtractFeaturesFunc",
    "PreprocessFunc",
    "analyse_behaviour",
    "classify_single",
    "combine_analysis",
    "distance",
    "dur_frames_from_likelihood",
    "extract_features",
    "format_video",
    "get_video_metadata",
    "in_roi",
    "interpolate",
    "interpolate_stationary",
    "ma_dlc_run_batch",
    "ma_dlc_run_single",
    "px_per_mm",
    "social_distance",
    "start_frame_from_csv",
    "start_frame_from_likelihood",
    "start_stop_trim",
    "stop_frame_from_dur",
    "stop_frame_from_likelihood",
]
