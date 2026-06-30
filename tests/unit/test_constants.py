"""Unit tests for constants module."""

from behavysis.constants import (
    ANALYSIS_DIR,
    CACHE_DIR,
    DF_IO_FORMAT,
    FileExts,
    Folders,
)


class TestFolders:
    """Tests for Folders enum."""

    def test_folders_values(self) -> None:
        """Folders should have expected string values."""
        assert Folders.CONFIG.value == "0_config"
        assert Folders.RAW_VID.value == "1_raw_videos"
        assert Folders.FORMATTED_VID.value == "2_formatted_videos"
        assert Folders.KEYPOINTS.value == "3_keypoints"
        assert Folders.PREPROCESSED.value == "4_preprocessed"
        assert Folders.FEATURES_EXTRACTED.value == "5_features_extracted"
        assert Folders.PREDICTED_BEHAVIOUR.value == "6_predicted_behaviour"
        assert Folders.SCORED_BEHAVIOUR.value == "7_scored_behaviour"
        assert Folders.ANALYSIS_COMBINED.value == "9_analysis_combined"

    def test_folders_are_ordered(self) -> None:
        """Folder names should start with numeric prefixes for ordering."""
        for folder in Folders:
            assert folder.value[0].isdigit()


class TestFileExts:
    """Tests for FileExts enum."""

    def test_video_extensions(self) -> None:
        """Video-related folders should have mp4 extension."""
        assert FileExts.RAW_VIDEO.value == "mp4"
        assert FileExts.FORMATTED_VIDEO.value == "mp4"
        assert FileExts.EVALUATE_VIDEO.value == "mp4"

    def test_config_extension(self) -> None:
        """Config folder should have json extension."""
        assert FileExts.CONFIG.value == "json"

    def test_data_extensions(self) -> None:
        """Data folders should use DF_IO_FORMAT."""
        assert FileExts.KEYPOINTS.value == DF_IO_FORMAT
        assert FileExts.PREPROCESSED.value == DF_IO_FORMAT
        assert FileExts.FEATURES_EXTRACTED.value == DF_IO_FORMAT
        assert FileExts.PREDICTED_BEHAVIOUR.value == DF_IO_FORMAT
        assert FileExts.SCORED_BEHAVIOUR.value == DF_IO_FORMAT
        assert FileExts.ANALYSIS_COMBINED.value == DF_IO_FORMAT


class TestConstants:
    """Tests for other constants."""

    def test_df_io_format(self) -> None:
        """DF_IO_FORMAT should be parquet."""
        assert DF_IO_FORMAT == "parquet"

    def test_analysis_dir(self) -> None:
        """ANALYSIS_DIR should be a Path with expected name."""
        assert ANALYSIS_DIR.name == "8_analysis"

    def test_cache_dir(self) -> None:
        """CACHE_DIR should be in home directory under .behavysis."""
        assert CACHE_DIR.name == ".behavysis"
        assert ".behavysis" in str(CACHE_DIR)
