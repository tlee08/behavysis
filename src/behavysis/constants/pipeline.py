"""Pipeline folder and file structure constants."""

from enum import Enum
from pathlib import Path

DF_IO_FORMAT = "parquet"


class Folders(Enum):
    """Enum for the pipeline folders."""

    CONFIG = "0_config"
    RAW_VID = "1_raw_video"
    FORMATTED_VID = "2_formatted_video"
    KEYPOINTS = "3_keypoints"
    PREPROCESSED = "4_preprocessed"
    FEATURES_EXTRACTED = "5_features_extracted"
    PREDICTED_BEHAVS = "6_predicted_behaviour"
    SCORED_BEHAVS = "7_scored_behaviour"
    ANALYSIS_COMBINED = "9_analysis_combined"
    EVALUATE_VID = "10_evaluate_video"


class FileExts(Enum):
    """Enum for file extensions by folder type."""

    CONFIG = "json"
    RAW_VIDEO = "mp4"
    FORMATTED_VIDEO = "mp4"
    KEYPOINTS = DF_IO_FORMAT
    PREPROCESSED = DF_IO_FORMAT
    FEATURES_EXTRACTED = DF_IO_FORMAT
    PREDICTED_BEHAVIOUR = DF_IO_FORMAT
    SCORED_BEHAVIOUR = DF_IO_FORMAT
    ANALYSIS_COMBINED = DF_IO_FORMAT
    EVALUATE_VIDEO = "mp4"


DEFAULT_CONFIG_FP = "default_config.json"

ANALYSIS_DIR = Path("8_analysis")
CACHE_DIR = Path.home() / ".behavysis"
