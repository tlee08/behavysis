"""Unit tests for ExperimentConfig and related models."""

from behavysis.funcs.extract_features.extract_generic import ExtractGenericConfig
from behavysis.models import ClassifierRef


class TestExtractGenericConfig:
    def test_individuals_and_bodyparts_required(self):
        cfg = ExtractGenericConfig(
            individuals=["mouse1marked", "mouse2unmarked"],
            bodyparts=["Nose", "LeftEar", "TailBase1"],
            angles=[],
        )
        assert cfg.individuals == ["mouse1marked", "mouse2unmarked"]
        assert cfg.bodyparts == ["Nose", "LeftEar", "TailBase1"]


class TestClassifierRef:
    def test_defaults(self):
        cfg = ClassifierRef()
        assert cfg.sub_behaviour == []
