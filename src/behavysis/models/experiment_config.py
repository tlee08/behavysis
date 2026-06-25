"""Experiment configuration models for the behavysis pipeline."""

from pydantic import BaseModel, ConfigDict

from behavysis.constants import (
    BPTS_CENTRE,
    BPTS_CORNERS,
    BPTS_FRONT,
    BPTS_SIMBA,
    INDIVS_SIMBA,
)

from .funcs.analyse import (
    AnalyseConfig,
    FreezingConfig,
    InRoiConfig,
    SocialDistanceConfig,
    SpeedConfig,
)
from .funcs.calculate_params import (
    CalculateParamsConfig,
    FromLikelihoodConfig,
)
from .funcs.classify_behavs import ClassifyBehavConfig
from .funcs.evaluate_vid import EvaluateVidConfig
from .funcs.extract_features import ExtractFeaturesConfig
from .funcs.format_vid import FormatVidConfig, VidMetadata
from .funcs.preprocess import PreprocessConfig, RefineIdsConfig
from .funcs.run_dlc import RunDlcConfig


class AnalysisConfig(BaseModel):
    """Validated analysis configuration parameters."""

    fps: float
    width_px: float
    height_px: float
    px_per_mm: float
    bins_sec: list
    custom_bins_sec: list


class UserConfig(BaseModel):
    """User Config."""

    format_vid: FormatVidConfig = FormatVidConfig()
    run_dlc: RunDlcConfig = RunDlcConfig()
    calculate_params: CalculateParamsConfig = CalculateParamsConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    extract_features: ExtractFeaturesConfig = ExtractFeaturesConfig()
    classify_behavs: list[ClassifyBehavConfig] = []
    analyse: AnalyseConfig = AnalyseConfig()
    evaluate_vid: EvaluateVidConfig = EvaluateVidConfig()


class AutoConfig(BaseModel):
    """Auto Config."""

    raw_vid: VidMetadata = VidMetadata()
    formatted_vid: VidMetadata = VidMetadata()

    px_per_mm: float = -1
    start_frame: int = -1
    stop_frame: int = -1
    dur_frames: int = -1


class RefConfig(BaseModel):
    """Ref Config."""

    model_config = ConfigDict(extra="allow")


class ExperimentConfig(BaseModel):
    """Experiment Config."""

    user: UserConfig = UserConfig()
    auto: AutoConfig = AutoConfig()
    ref: RefConfig = RefConfig()

    def get_ref[T](self, val: T | str) -> T:
        """Resolve reference values from the ref section.

        If val is in reference format (`"--<ref_name>"`), returns the
        referenced value from the ref section. Otherwise returns val unchanged.

        Parameters
        ----------
        val : Any
            Value to resolve, potentially a reference string.

        Returns:
        -------
        Any
            Resolved value or original val if not a reference.
        """
        # Check if the value is in the reference format
        if isinstance(val, str) and val.startswith("--"):
            val_str = str(val)
            # Remove the '--' from the val
            val_str_ref = val_str[2:]
            # Check if the value exists in the reference store
            assert hasattr(self.ref, val_str_ref), (
                f"Value '{val_str_ref}' can't be found in the config reference section."
            )
            return getattr(self.ref, val_str_ref)
        # Otherwise, return value itself
        return val  # ty:ignore[invalid-return-type]

    def get_analysis_config(self) -> "AnalysisConfig":
        """Get validated analysis configuration parameters.

        Returns:
        -------
        AnalysisConfig
            Pydantic model containing fps, dimensions, scale, and bin sizes.

        Raises:
        ------
        AssertionError
            If required video metadata or px_per_mm not set.
        """
        assert self.auto.formatted_vid.fps > 0
        assert self.auto.formatted_vid.width_px > 0
        assert self.auto.formatted_vid.height_px > 0
        assert self.auto.px_per_mm > 0
        return AnalysisConfig(
            fps=float(self.auto.formatted_vid.fps),
            width_px=float(self.auto.formatted_vid.width_px),
            height_px=float(self.auto.formatted_vid.height_px),
            px_per_mm=float(self.auto.px_per_mm),
            bins_sec=list(self.get_ref(self.user.analyse.bins_sec)),
            custom_bins_sec=list(self.get_ref(self.user.analyse.custom_bins_sec)),
        )


def get_default_config() -> ExperimentConfig:
    """Get default config."""
    return ExperimentConfig(
        user=UserConfig(
            format_vid=FormatVidConfig(width_px=960, height_px=540, fps=15),
            calculate_params=CalculateParamsConfig(
                from_likelihood=FromLikelihoodConfig(bodyparts="--bpts_simba")
            ),
            preprocess=PreprocessConfig(
                refine_ids=RefineIdsConfig(bodyparts="--bpts_centre")
            ),
            extract_features=ExtractFeaturesConfig(
                individuals="--indivs_simba", bodyparts="--bpts_simba"
            ),
            classify_behavs=[ClassifyBehavConfig()],
            analyse=AnalyseConfig(
                in_roi=[
                    InRoiConfig(roi_corners="--bpts_corners", bodyparts="--bpts_front")
                ],
                speed=SpeedConfig(bodyparts="--bpts_centre"),
                social_distance=SocialDistanceConfig(bodyparts="--bpts_centre"),
                freezing=FreezingConfig(bodyparts="--bpts_centre"),
            ),
        ),
        ref=RefConfig.model_validate(
            {
                "indivs_simba": INDIVS_SIMBA,
                "bpts_simba": BPTS_SIMBA,
                "bpts_centre": BPTS_CENTRE,
                "bpts_front": BPTS_FRONT,
                "bpts_corners": BPTS_CORNERS,
            }
        ),
    )
