"""Pipeline folder and file structure constants."""

from enum import Enum
from pathlib import Path

DF_IO_FORMAT = "parquet"


class Folders(Enum):
    """Enum for the pipeline folders."""

    CONFIGS = "0_configs"
    RAW_VID = "1_raw_vid"
    FORMATTED_VID = "2_formatted_vid"
    KEYPOINTS = "3_keypoints"
    PREPROCESSED = "4_preprocessed"
    FEATURES_EXTRACTED = "5_features_extracted"
    PREDICTED_BEHAVS = "6_predicted_behavs"
    SCORED_BEHAVS = "7_scored_behavs"
    ANALYSIS_COMBINED = "9_analysis_combined"
    EVALUATE_VID = "10_evaluate_vid"


class FileExts(Enum):
    """Enum for file extensions by folder type."""

    CONFIGS = "json"
    RAW_VID = "mp4"
    FORMATTED_VID = "mp4"
    KEYPOINTS = DF_IO_FORMAT
    PREPROCESSED = DF_IO_FORMAT
    FEATURES_EXTRACTED = DF_IO_FORMAT
    PREDICTED_BEHAVS = DF_IO_FORMAT
    SCORED_BEHAVS = DF_IO_FORMAT
    ANALYSIS_COMBINED = DF_IO_FORMAT
    EVALUATE_VID = "mp4"


DIAGNOSTICS_DIR = Path("0_diagnostics")
ANALYSIS_DIR = Path("8_analysis")
CACHE_DIR = Path.home() / ".behavysis"
