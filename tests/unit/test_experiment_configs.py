"""Unit tests for ExperimentConfig and related models."""

import pytest

from behavysis.models import (
    ClassifyBehaviourConfig,
    ExtractFeaturesConfig,
)


class TestExtractFeaturesConfig:
    def test_individuals_and_bodyparts_required(self):
        cfg = ExtractFeaturesConfig(
            individuals=["mouse1marked", "mouse2unmarked"],
            bodyparts=["Nose", "LeftEar", "TailBase1"],
        )
        assert cfg.individuals == ["mouse1marked", "mouse2unmarked"]
        assert cfg.bodyparts == ["Nose", "LeftEar", "TailBase1"]

    def test_validate_bodypart_match_ok(self):
        a = ExtractFeaturesConfig(
            individuals=["mouse1", "mouse2"],
            bodyparts=["Nose", "TailBase1"],
        )
        b = ExtractFeaturesConfig(
            individuals=["mouse1", "mouse2"],
            bodyparts=["Nose", "TailBase1"],
        )
        a.validate_bodypart_match(b)

    def test_validate_bodypart_match_individuals_mismatch(self):
        a = ExtractFeaturesConfig(
            individuals=["mouse1"],
            bodyparts=["Nose", "TailBase1"],
        )
        b = ExtractFeaturesConfig(
            individuals=["mouse2"],
            bodyparts=["Nose", "TailBase1"],
        )
        with pytest.raises(ValueError, match="Individual mismatch"):
            a.validate_bodypart_match(b)

    def test_validate_bodypart_match_bodyparts_mismatch(self):
        a = ExtractFeaturesConfig(
            individuals=["mouse1"],
            bodyparts=["Nose"],
        )
        b = ExtractFeaturesConfig(
            individuals=["mouse1"],
            bodyparts=["TailBase1"],
        )
        with pytest.raises(ValueError, match="Bodypart mismatch"):
            a.validate_bodypart_match(b)


class TestClassifyBehaviourConfig:
    def test_required_fields(self):
        cfg = ClassifyBehaviourConfig(
            individuals=["mouse1", "mouse2"],
            bodyparts=["Nose", "TailBase1"],
        )
        assert cfg.individuals == ["mouse1", "mouse2"]
        assert cfg.bodyparts == ["Nose", "TailBase1"]
        assert cfg.model_type == "rf"

    def test_defaults(self):
        cfg = ClassifyBehaviourConfig(
            individuals=["mouse1"],
            bodyparts=["Nose"],
        )
        assert cfg.behaviour_name == "behaviour_name"
        assert cfg.min_empty_window_secs == 0.2
        assert cfg.pcutoff is None
