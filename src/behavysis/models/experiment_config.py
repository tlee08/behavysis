"""Experiment configuration models for the behavysis pipeline."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

from ._validators import ConfigNotConfiguredError

# ═══════════════════════════════════════════════════════════════════════════════
# Config Sub Models
# ═══════════════════════════════════════════════════════════════════════════════


class SubfuncModel(BaseModel):
    """SubfuncModel."""

    model_config = ConfigDict(extra="allow")

    def require[T: BaseModel](self, name: str, model_cls: type[T]) -> T:
        """Require and validate a single sub-config by function name."""
        # Check that subconfig exists
        if not hasattr(self, name):
            msg = f"analyse.{name}"
            raise ConfigNotConfiguredError(msg)
        # Check that subconfig is of the correct type
        value_key = getattr(self, name)
        # Return subconfig
        return model_cls.model_validate(value_key)

    def require_list[T: BaseModel](self, name: str, model_cls: type[T]) -> list[T]:
        """Require and validate a list sub-config (e.g. in_roi)."""
        # Check that subconfig exists
        if not hasattr(self, name):
            msg = f"analyse.{name}"
            raise ConfigNotConfiguredError(msg)
        value_key_ls = getattr(self, name)
        return [model_cls.model_validate(value_key) for value_key in value_key_ls]


class FormatVideoConfig(BaseModel):
    """FormatVidConfig."""

    width_px: PositiveInt | None = None
    height_px: PositiveInt | None = None
    fps: PositiveFloat | None = None
    start_sec: PositiveFloat | None = None
    stop_sec: PositiveFloat | None = None


class RunDlcConfig(BaseModel):
    """RunDlcConfig."""

    model_fp: Path = Path("path") / "to" / "DEEPLABCUT_model" / "config.yaml"


class CalculateParametersConfig(SubfuncModel):
    """CalculateParametersConfig."""


class PreprocessConfig(SubfuncModel):
    """PreprocessConfig."""


class ExtractFeaturesConfig(BaseModel):
    """Configuration for generic feature extraction.

    Specifies which individuals and bodyparts to include. Features are
    computed programmatically from the full cartesian product — no
    semantic bodypart roles required.
    """

    individuals: list[str]
    bodyparts: list[str]

    def validate_bodypart_match(self, other: ExtractFeaturesConfig) -> None:
        """Validate that another config has matching individuals and bodyparts."""
        if set(self.individuals) != set(other.individuals):
            msg = (
                f"Individual mismatch: {sorted(self.individuals)} vs "
                f"{sorted(other.individuals)}"
            )
            raise ValueError(msg)
        if set(self.bodyparts) != set(other.bodyparts):
            msg = (
                f"Bodypart mismatch: {sorted(self.bodyparts)} vs "
                f"{sorted(other.bodyparts)}"
            )
            raise ValueError(msg)


class ClassifyBehaviourConfig(BaseModel):
    """ClassifyBehaviourConfig.

    Specifies which trained model to use for behaviour classification.
    ``individuals`` and ``bodyparts`` must match the experiment's
    extract_features config — validated at classify time.
    """

    proj_dir: Path = Path("path") / "to" / "project_dir"
    behaviour_name: str = "behaviour_name"
    model_type: str = "rf"
    individuals: list[str]
    bodyparts: list[str]
    pcutoff: PositiveFloat | None = None
    min_empty_window_secs: PositiveFloat = 0.2
    user_defined: list[str] = []


class AnalyseConfig(SubfuncModel):
    """AnalyseConfig."""

    bins_sec_ls: list[PositiveInt] = [30, 60, 120]
    custom_bins_sec_ls: list[PositiveInt] = [60, 120, 300, 600]


# ═══════════════════════════════════════════════════════════════════════════════
# Main Config Model
# ═══════════════════════════════════════════════════════════════════════════════


class ExperimentConfig(BaseModel):
    """Experiment Config."""

    format_video: FormatVideoConfig | None
    run_dlc: RunDlcConfig | None
    calculate_parameters: CalculateParametersConfig | None
    preprocess: PreprocessConfig | None
    extract_features: ExtractFeaturesConfig | None
    classify_behaviour: list[ClassifyBehaviourConfig] | None
    analyse: AnalyseConfig | None

    @classmethod
    def read_yaml(cls, fp: Path) -> ExperimentConfig:
        """Read the config from a yaml file."""
        return ExperimentConfig.model_validate(
            yaml.safe_load(fp.open("r")),
        )

    def require_format_video(self) -> FormatVideoConfig:
        """Require the format_video config."""
        if self.format_video is None:
            msg = "format_video"
            raise ConfigNotConfiguredError(msg)
        return self.format_video

    def require_run_dlc(self) -> RunDlcConfig:
        """Require the run_dlc config."""
        if self.run_dlc is None:
            msg = "run_dlc"
            raise ConfigNotConfiguredError(msg)
        return self.run_dlc

    def require_calculate_parameters(self) -> CalculateParametersConfig:
        """Require the calculate_parameters config."""
        if self.calculate_parameters is None:
            msg = "calculate_parameters"
            raise ConfigNotConfiguredError(msg)
        return self.calculate_parameters

    def require_preprocess(self) -> PreprocessConfig:
        """Require the preprocess config."""
        if self.preprocess is None:
            msg = "preprocess"
            raise ConfigNotConfiguredError(msg)
        return self.preprocess

    def require_extract_features(self) -> ExtractFeaturesConfig:
        """Require the extract_features config."""
        if self.extract_features is None:
            msg = "extract_features"
            raise ConfigNotConfiguredError(msg)
        return self.extract_features

    def require_classify_behaviour(self) -> list[ClassifyBehaviourConfig]:
        """Require the classify_behaviour config."""
        if self.classify_behaviour is None:
            msg = "classify_behaviour"
            raise ConfigNotConfiguredError(msg)
        return self.classify_behaviour

    def require_analyse(self) -> AnalyseConfig:
        """Require the analyse config."""
        if self.analyse is None:
            msg = "analyse"
            raise ConfigNotConfiguredError(msg)
        return self.analyse
