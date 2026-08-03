"""Unit tests for constants module."""

from behavysis.constants import ANALYSIS_DIR, CACHE_DIR, DF_IO_FORMAT


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
