"""Pipeline folder and file structure constants."""

from pathlib import Path

DF_IO_FORMAT = "parquet"


CONFIG_DIR = "0_config"
METADATA_DIR = "0_metadata"
RAW_VIDEO_DIR = "1_raw_videos"
FORMATTED_VIDEO_DIR = "2_formatted_videos"
KEYPOINTS_DIR = "3_keypoints"
PREPROCESSED_DIR = "4_preprocessed"
FEATURES_EXTRACTED_DIR = "5_features_extracted"
BEHAVIOUR_PREDICTED_DIR = "6_behaviour_predicted"
BEHAVIOUR_SCORED_DIR = "7_behaviour_scored"
ANALYSIS_COMBINED_DIR = "9_analysis_combined"

STAGES = {
    CONFIG_DIR: "yaml",
    METADATA_DIR: "json",
    RAW_VIDEO_DIR: "mp4",
    FORMATTED_VIDEO_DIR: "mp4",
    KEYPOINTS_DIR: DF_IO_FORMAT,
    PREPROCESSED_DIR: DF_IO_FORMAT,
    FEATURES_EXTRACTED_DIR: DF_IO_FORMAT,
    BEHAVIOUR_PREDICTED_DIR: DF_IO_FORMAT,
    BEHAVIOUR_SCORED_DIR: DF_IO_FORMAT,
    ANALYSIS_COMBINED_DIR: DF_IO_FORMAT,
}


DEFAULT_CONFIG_FP = "default_config.yaml"

ANALYSIS_DIR = Path("8_analysis")
CACHE_DIR = Path.home() / ".behavysis"
