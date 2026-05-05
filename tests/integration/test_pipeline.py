"""Integration tests for the behavysis pipeline.

These tests require:
1. Sample video files
2. Sample keypoints parquet files (generated from DeepLabCut)
3. Proper config files

Run with: pytest -m integration
"""

import pytest


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests that run the full pipeline."""

    @pytest.mark.skip(reason="Requires sample video and keypoints data")
    def test_full_pipeline_run(self) -> None:
        """Test running the full pipeline on sample data."""
        pass

    @pytest.mark.skip(reason="Requires sample keypoints data")
    def test_start_frame_detection(self) -> None:
        """Test start frame detection with real keypoints data."""
        pass

    @pytest.mark.skip(reason="Requires sample keypoints data")
    def test_duration_calculation(self) -> None:
        """Test duration calculation with real keypoints data."""
        pass
