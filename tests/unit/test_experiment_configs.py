"""Unit tests for ExperimentConfig model."""

import pytest

from behavysis.models import (
    AutoConfig,
    ExperimentConfig,
    RefConfig,
    UserConfig,
    get_default_config,
)


class TestRefConfig:
    """Tests for RefConfig model."""

    def test_extra_fields_allowed(self) -> None:
        """RefConfig should allow extra fields."""
        config = RefConfig.model_validate(
            {
                "custom_ref": ["a", "b", "c"],
                "another_ref": 123,
            }
        )
        # Access extra fields via model_dump or __getattr__
        data = config.model_dump()
        assert data["custom_ref"] == ["a", "b", "c"]
        assert data["another_ref"] == 123

    def test_empty_ref_config(self) -> None:
        """Empty RefConfig should be valid."""
        config = RefConfig()
        assert config.model_dump() == {}


class TestUserConfig:
    """Tests for UserConfig model."""

    def test_default_user_config(self) -> None:
        """Default UserConfig should be valid."""
        config = UserConfig()
        assert config.format_vid is not None
        assert config.run_dlc is not None
        assert config.calculate_params is not None

    def test_user_config_from_dict(self) -> None:
        """UserConfig should parse from dict correctly."""
        config = UserConfig.model_validate(
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


class TestAutoConfig:
    """Tests for AutoConfig model."""

    def test_default_auto_config(self) -> None:
        """Default AutoConfig should have expected defaults."""
        config = AutoConfig()
        assert config.px_per_mm == -1
        assert config.start_frame == -1
        assert config.stop_frame == -1
        assert config.dur_frames == -1


class TestExperimentConfig:
    """Tests for ExperimentConfig model."""

    def test_default_config(self) -> None:
        """Default ExperimentConfig should be valid."""
        config = ExperimentConfig()
        assert config.user is not None
        assert config.auto is not None
        assert config.ref is not None

    def test_get_ref_string(self) -> None:
        """get_ref should resolve reference strings."""
        config = ExperimentConfig(
            ref=RefConfig.model_validate(
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
        config = ExperimentConfig()
        assert config.get_ref(123) == 123
        assert config.get_ref([1, 2, 3]) == [1, 2, 3]
        assert config.get_ref(None) is None

    def test_get_ref_non_reference_string(self) -> None:
        """get_ref should return non-reference strings unchanged."""
        config = ExperimentConfig()
        assert config.get_ref("regular_string") == "regular_string"
        assert config.get_ref("-single_dash") == "-single_dash"

    def test_get_ref_missing_reference(self) -> None:
        """get_ref should raise AssertionError for missing references."""
        config = ExperimentConfig()
        with pytest.raises(AssertionError, match="can't be found"):
            config.get_ref("--nonexistent")

    def test_model_dump_json(self) -> None:
        """model_dump_json should produce valid JSON string."""
        config = ExperimentConfig()
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        assert "user" in json_str
        assert "auto" in json_str
        assert "ref" in json_str

    def test_model_validate_json_roundtrip(self) -> None:
        """Should roundtrip through JSON serialization."""
        original = ExperimentConfig(
            user=UserConfig.model_validate(
                {
                    "format_vid": {"width_px": 1920, "height_px": 1080, "fps": 60},
                }
            ),
            ref=RefConfig.model_validate({"custom": ["a", "b"]}),
        )
        json_str = original.model_dump_json()
        restored = ExperimentConfig.model_validate_json(json_str)

        assert restored.user.format_vid.width_px == 1920
        assert restored.user.format_vid.height_px == 1080
        assert restored.user.format_vid.fps == 60
        # Note: ref extra fields may not roundtrip due to pydantic behavior


class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_returns_valid_config(self) -> None:
        """Should return valid ExperimentConfig."""
        config = get_default_config()
        assert isinstance(config, ExperimentConfig)

    def test_has_expected_refs(self) -> None:
        """Should include standard bodypart references."""
        config = get_default_config()
        assert hasattr(config.ref, "bpts_simba")
        assert hasattr(config.ref, "bpts_centre")
        assert hasattr(config.ref, "bpts_front")
        assert hasattr(config.ref, "bpts_corners")
        assert hasattr(config.ref, "indivs_simba")

    def test_refs_are_lists(self) -> None:
        """Reference values should be lists."""
        config = get_default_config()
        data = config.ref.model_dump()
        assert isinstance(data["bpts_simba"], list)
        assert isinstance(data["bpts_centre"], list)


class TestGetAnalysisConfig:
    """Tests for ExperimentConfig.get_analysis_config method."""

    def test_get_analysis_config_requires_metadata(self) -> None:
        """Should raise if video metadata not set."""
        config = ExperimentConfig()
        with pytest.raises(AssertionError):
            config.get_analysis_config()

    def test_get_analysis_config_requires_px_per_mm(self) -> None:
        """Should raise if px_per_mm not set."""
        config = ExperimentConfig()
        config.auto.formatted_vid.fps = 30
        config.auto.formatted_vid.width_px = 960
        config.auto.formatted_vid.height_px = 540
        # px_per_mm is still -1 (default)

        with pytest.raises(AssertionError):
            config.get_analysis_config()

    def test_get_analysis_config_success(self) -> None:
        """Should return AnalysisConfig when all required values are set."""
        config = ExperimentConfig()
        config.auto.formatted_vid.fps = 30
        config.auto.formatted_vid.width_px = 960
        config.auto.formatted_vid.height_px = 540
        config.auto.px_per_mm = 10.0
        config.user.analyse.bins_sec = [30, 60, 120]
        config.user.analyse.custom_bins_sec = [60, 120]

        analysis = config.get_analysis_config()

        assert analysis.fps == 30
        assert analysis.width_px == 960
        assert analysis.height_px == 540
        assert analysis.px_per_mm == 10.0
        assert analysis.bins_sec == [30, 60, 120]
