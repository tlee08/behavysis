"""Pipeline folder and file structure constants."""

from pathlib import Path

DF_IO_FORMAT = "parquet"


CONFIG_DIR = "0_config"
RAW_VIDEO_DIR = "1_raw_videos"
FORMATTED_VIDEO_DIR = "2_formatted_videos"
KEYPOINTS_DIR = "3_keypoints"
PREPROCESSED_DIR = "4_preprocessed"
FEATURES_EXTRACTED_DIR = "5_features_extracted"
PREDICTED_BEHAVIOUR_DIR = "6_predicted_behaviour"
SCORED_BEHAVIOUR_DIR = "7_scored_behaviour"
ANALYSIS_COMBINED_DIR = "9_analysis_combined"

STAGES = {
    CONFIG_DIR: "json",
    RAW_VIDEO_DIR: "mp4",
    FORMATTED_VIDEO_DIR: "mp4",
    KEYPOINTS_DIR: DF_IO_FORMAT,
    PREPROCESSED_DIR: DF_IO_FORMAT,
    FEATURES_EXTRACTED_DIR: DF_IO_FORMAT,
    PREDICTED_BEHAVIOUR_DIR: DF_IO_FORMAT,
    SCORED_BEHAVIOUR_DIR: DF_IO_FORMAT,
    ANALYSIS_COMBINED_DIR: DF_IO_FORMAT,
}


DEFAULT_CONFIG_FP = "default_config.json"

ANALYSIS_DIR = Path("8_analysis")
CACHE_DIR = Path.home() / ".behavysis"
