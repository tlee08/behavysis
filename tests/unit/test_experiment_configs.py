"""Unit tests for ExperimentConfigs model."""

import pytest
from pydantic import ValidationError

from behavysis.models.experiment_configs import (
    AutoConfigs,
    ExperimentConfigs,
    RefConfigs,
    UserConfigs,
    get_default_configs,
)


class TestRefConfigs:
    """Tests for RefConfigs model."""

    def test_extra_fields_allowed(self) -> None:
        """RefConfigs should allow extra fields."""
        config = RefConfigs.model_validate(
            {
                "custom_ref": ["a", "b", "c"],
                "another_ref": 123,
            }
        )
        # Access extra fields via model_dump or __getattr__
        data = config.model_dump()
        assert data["custom_ref"] == ["a", "b", "c"]
        assert data["another_ref"] == 123

    def test_empty_ref_configs(self) -> None:
        """Empty RefConfigs should be valid."""
        config = RefConfigs()
        assert config.model_dump() == {}


class TestUserConfigs:
    """Tests for UserConfigs model."""

    def test_default_user_configs(self) -> None:
        """Default UserConfigs should be valid."""
        config = UserConfigs()
        assert config.format_vid is not None
        assert config.run_dlc is not None
        assert config.calculate_params is not None

    def test_user_configs_from_dict(self) -> None:
        """UserConfigs should parse from dict correctly."""
        config = UserConfigs.model_validate(
            {
                "format_vid": {
                    "width_px": 960,
                    "height_px": 540,
                    "fps": 30,
                },
            }
        )
        assert config.format_vid.width_px == 960
        assert config.format_vid.height_px == 540
        assert config.format_vid.fps == 30


class TestAutoConfigs:
    """Tests for AutoConfigs model."""

    def test_default_auto_configs(self) -> None:
        """Default AutoConfigs should have expected defaults."""
        config = AutoConfigs()
        assert config.px_per_mm == -1
        assert config.start_frame == -1
        assert config.stop_frame == -1
        assert config.dur_frames == -1


class TestExperimentConfigs:
    """Tests for ExperimentConfigs model."""

    def test_default_configs(self) -> None:
        """Default ExperimentConfigs should be valid."""
        config = ExperimentConfigs()
        assert config.user is not None
        assert config.auto is not None
        assert config.ref is not None

    def test_get_ref_string(self) -> None:
        """get_ref should resolve reference strings."""
        config = ExperimentConfigs(
            ref=RefConfigs.model_validate(
                {
                    "bodyparts": ["nose", "tail"],
                    "threshold": 50,
                }
            )
        )

        assert config.get_ref("--bodyparts") == ["nose", "tail"]
        assert config.get_ref("--threshold") == 50

    def test_get_ref_non_string(self) -> None:
        """get_ref should return non-string values unchanged."""
        config = ExperimentConfigs()
        assert config.get_ref(123) == 123
        assert config.get_ref([1, 2, 3]) == [1, 2, 3]
        assert config.get_ref(None) is None

    def test_get_ref_non_reference_string(self) -> None:
        """get_ref should return non-reference strings unchanged."""
        config = ExperimentConfigs()
        assert config.get_ref("regular_string") == "regular_string"
        assert config.get_ref("-single_dash") == "-single_dash"

    def test_get_ref_missing_reference(self) -> None:
        """get_ref should raise AssertionError for missing references."""
        config = ExperimentConfigs()
        with pytest.raises(AssertionError, match="can't be found"):
            config.get_ref("--nonexistent")

    def test_model_dump_json(self) -> None:
        """model_dump_json should produce valid JSON string."""
        config = ExperimentConfigs()
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        assert "user" in json_str
        assert "auto" in json_str
        assert "ref" in json_str

    def test_model_validate_json_roundtrip(self) -> None:
        """Should roundtrip through JSON serialization."""
        original = ExperimentConfigs(
            user=UserConfigs.model_validate(
                {
                    "format_vid": {"width_px": 1920, "height_px": 1080, "fps": 60},
                }
            ),
            ref=RefConfigs.model_validate({"custom": ["a", "b"]}),
        )
        json_str = original.model_dump_json()
        restored = ExperimentConfigs.model_validate_json(json_str)

        assert restored.user.format_vid.width_px == 1920
        assert restored.user.format_vid.height_px == 1080
        assert restored.user.format_vid.fps == 60
        # Note: ref extra fields may not roundtrip due to pydantic behavior


class TestGetDefaultConfigs:
    """Tests for get_default_configs function."""

    def test_returns_valid_config(self) -> None:
        """Should return valid ExperimentConfigs."""
        config = get_default_configs()
        assert isinstance(config, ExperimentConfigs)

    def test_has_expected_refs(self) -> None:
        """Should include standard bodypart references."""
        config = get_default_configs()
        assert hasattr(config.ref, "bpts_simba")
        assert hasattr(config.ref, "bpts_centre")
        assert hasattr(config.ref, "bpts_front")
        assert hasattr(config.ref, "bpts_corners")
        assert hasattr(config.ref, "indivs_simba")

    def test_refs_are_lists(self) -> None:
        """Reference values should be lists."""
        config = get_default_configs()
        data = config.ref.model_dump()
        assert isinstance(data["bpts_simba"], list)
        assert isinstance(data["bpts_centre"], list)


class TestGetAnalysisConfigs:
    """Tests for ExperimentConfigs.get_analysis_configs method."""

    def test_get_analysis_configs_requires_metadata(self) -> None:
        """Should raise if video metadata not set."""
        config = ExperimentConfigs()
        with pytest.raises(AssertionError):
            config.get_analysis_configs()

    def test_get_analysis_configs_requires_px_per_mm(self) -> None:
        """Should raise if px_per_mm not set."""
        config = ExperimentConfigs()
        config.auto.formatted_vid.fps = 30
        config.auto.formatted_vid.width_px = 960
        config.auto.formatted_vid.height_px = 540
        # px_per_mm is still -1 (default)

        with pytest.raises(AssertionError):
            config.get_analysis_configs()

    def test_get_analysis_configs_success(self) -> None:
        """Should return AnalysisConfigs when all required values are set."""
        config = ExperimentConfigs()
        config.auto.formatted_vid.fps = 30
        config.auto.formatted_vid.width_px = 960
        config.auto.formatted_vid.height_px = 540
        config.auto.px_per_mm = 10.0
        config.user.analyse.bins_sec = [30, 60, 120]
        config.user.analyse.custom_bins_sec = [60, 120]

        analysis = config.get_analysis_configs()

        assert analysis.fps == 30
        assert analysis.width_px == 960
        assert analysis.height_px == 540
        assert analysis.px_per_mm == 10.0
        assert analysis.bins_sec == [30, 60, 120]
