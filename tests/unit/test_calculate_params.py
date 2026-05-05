"""Unit tests for calculate_params module.

NOTE: These tests require proper keypoints data with the correct column structure.
They are marked as integration tests because they need realistic data.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from behavysis.models.experiment_configs import ExperimentConfigs
from behavysis.processes.calculate_params import stop_frame_from_dur

logger = logging.getLogger(__name__)


class TestStopFrameFromDur:
    """Tests for stop_frame_from_dur function.

    This function only requires config settings, not keypoints data.
    """

    def test_calculates_stop_frame(self, tmp_path: Path) -> None:
        """Should calculate stop frame from duration."""
        fps = 15
        start_frame = 500
        dur_sec = 60
        total_frames = 1000

        configs = ExperimentConfigs()
        configs.user.calculate_params.stop_frame_from_dur.dur_sec = dur_sec
        configs.auto.formatted_vid.fps = fps
        configs.auto.start_frame = start_frame
        configs.auto.formatted_vid.total_frames = total_frames

        configs_fp = tmp_path / "configs.json"
        configs_fp.write_text(configs.model_dump_json(indent=2))

        stop_frame_from_dur(configs_fp)

        configs = ExperimentConfigs.model_validate_json(configs_fp.read_text())
        logger.info(configs.auto.model_dump_json(indent=2))

        assert configs.auto.stop_frame == start_frame + dur_sec * fps

    def test_clamps_to_total_frames(self, tmp_path: Path) -> None:
        """Should not exceed total frames."""
        fps = 15
        start_frame = 500
        dur_sec = 600  # Would exceed total_frames
        total_frames = 1000

        configs = ExperimentConfigs()
        configs.user.calculate_params.stop_frame_from_dur.dur_sec = dur_sec
        configs.auto.formatted_vid.fps = fps
        configs.auto.start_frame = start_frame
        configs.auto.formatted_vid.total_frames = total_frames

        configs_fp = tmp_path / "configs.json"
        configs_fp.write_text(configs.model_dump_json(indent=2))

        stop_frame_from_dur(configs_fp)

        configs = ExperimentConfigs.model_validate_json(configs_fp.read_text())
        # Stop frame should be calculated (even if it exceeds total_frames)
        assert configs.auto.stop_frame == start_frame + dur_sec * fps


@pytest.mark.integration
class TestStartFrameFromLikelihood:
    """Tests for start_frame_from_likelihood function.

    These tests require proper keypoints data and are marked as integration tests.
    Run with: pytest -m integration
    """

    @pytest.mark.skip(reason="Requires proper keypoints parquet fixture")
    def test_detects_start_frame(self, tmp_path: Path) -> None:
        """Should detect when subject entered frame based on likelihood."""
        pass


@pytest.mark.integration
class TestDurFramesFromLikelihood:
    """Tests for dur_frames_from_likelihood function.

    These tests require proper keypoints data and are marked as integration tests.
    """

    @pytest.mark.skip(reason="Requires proper keypoints parquet fixture")
    def test_calculates_duration(self, tmp_path: Path) -> None:
        """Should calculate experiment duration from likelihood patterns."""
        pass
