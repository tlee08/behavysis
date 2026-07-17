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
            angles=[],
        )
        assert cfg.individuals == ["mouse1marked", "mouse2unmarked"]
        assert cfg.bodyparts == ["Nose", "LeftEar", "TailBase1"]


class TestClassifyBehaviourConfig:
    def test_defaults(self):
        cfg = ClassifyBehaviourConfig()
        assert cfg.sub_behaviour == []
