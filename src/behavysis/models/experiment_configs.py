"""Experiment configuration models for the behavysis pipeline."""

from typing import Any

from pydantic import BaseModel, ConfigDict

from behavysis.constants import (
    BPTS_CENTRE,
    BPTS_CORNERS,
    BPTS_FRONT,
    BPTS_SIMBA,
    INDIVS_SIMBA,
)
from behavysis.models.processes.analyse import (
    AnalyseConfigs,
    FreezingConfigs,
    InRoiConfigs,
    SocialDistanceConfigs,
    SpeedConfigs,
)
from behavysis.models.processes.calculate_params import (
    CalculateParamsConfigs,
    FromLikelihoodConfigs,
)
from behavysis.models.processes.classify_behavs import ClassifyBehavConfigs
from behavysis.models.processes.evaluate_vid import EvaluateVidConfigs
from behavysis.models.processes.extract_features import ExtractFeaturesConfigs
from behavysis.models.processes.format_vid import FormatVidConfigs, VidMetadata
from behavysis.models.processes.preprocess import PreprocessConfigs, RefineIdsConfigs
from behavysis.models.processes.run_dlc import RunDlcConfigs


class AnalysisConfigs(BaseModel):
    """Validated analysis configuration parameters."""

    fps: float
    width_px: float
    height_px: float
    px_per_mm: float
    bins_sec: list
    custom_bins_sec: list


class UserConfigs(BaseModel):
    format_vid: FormatVidConfigs = FormatVidConfigs()
    run_dlc: RunDlcConfigs = RunDlcConfigs()
    calculate_params: CalculateParamsConfigs = CalculateParamsConfigs()
    preprocess: PreprocessConfigs = PreprocessConfigs()
    extract_features: ExtractFeaturesConfigs = ExtractFeaturesConfigs()
    classify_behavs: list[ClassifyBehavConfigs] = []
    analyse: AnalyseConfigs = AnalyseConfigs()
    evaluate_vid: EvaluateVidConfigs = EvaluateVidConfigs()


class AutoConfigs(BaseModel):
    raw_vid: VidMetadata = VidMetadata()
    formatted_vid: VidMetadata = VidMetadata()

    px_per_mm: float = -1
    start_frame: int = -1
    stop_frame: int = -1
    dur_frames: int = -1

    @classmethod
    def get_field_names(cls) -> list[tuple[str, ...]]:
        """Returns the nested field names of the class as a list of tuples."""
        fields = []
        for name, type_ in cls.__annotations__.items():
            if hasattr(type_, "__annotations__"):
                for subfield in type_.get_field_names():
                    fields.append((name, *subfield))
            else:
                fields.append((name,))
        return fields


class RefConfigs(BaseModel):
    model_config = ConfigDict(extra="allow")


class ExperimentConfigs(BaseModel):
    user: UserConfigs = UserConfigs()
    auto: AutoConfigs = AutoConfigs()
    ref: RefConfigs = RefConfigs()

    def get_ref(self, val: Any) -> Any:
        """Resolve reference values from the ref section.

        If val is in reference format (`"--<ref_name>"`), returns the
        referenced value from the ref section. Otherwise returns val unchanged.

        Parameters
        ----------
        val : Any
            Value to resolve, potentially a reference string.

        Returns
        -------
        Any
            Resolved value or original val if not a reference.
        """
        # Check if the value is in the reference format
        if isinstance(val, str) and val.startswith("--"):
            # Remove the '--' from the val
            val = val[2:]
            # Check if the value exists in the reference store
            assert hasattr(self.ref, val), (
                f"Value '{val}' can't be found in the configs reference section."
            )
            return getattr(self.ref, val)
        return val

    def get_analysis_configs(self) -> "AnalysisConfigs":
        """Get validated analysis configuration parameters.

        Returns
        -------
        AnalysisConfigs
            Pydantic model containing fps, dimensions, scale, and bin sizes.

        Raises
        ------
        AssertionError
            If required video metadata or px_per_mm not set.
        """
        assert self.auto.formatted_vid.fps > 0
        assert self.auto.formatted_vid.width_px > 0
        assert self.auto.formatted_vid.height_px > 0
        assert self.auto.px_per_mm > 0
        return AnalysisConfigs(
            fps=float(self.auto.formatted_vid.fps),
            width_px=float(self.auto.formatted_vid.width_px),
            height_px=float(self.auto.formatted_vid.height_px),
            px_per_mm=float(self.auto.px_per_mm),
            bins_sec=list(self.get_ref(self.user.analyse.bins_sec)),
            custom_bins_sec=list(self.get_ref(self.user.analyse.custom_bins_sec)),
        )


def get_default_configs() -> ExperimentConfigs:
    return ExperimentConfigs(
        user=UserConfigs(
            format_vid=FormatVidConfigs(width_px=960, height_px=540, fps=15),
            calculate_params=CalculateParamsConfigs(
                from_likelihood=FromLikelihoodConfigs(bodyparts="--bpts_simba")
            ),
            preprocess=PreprocessConfigs(
                refine_ids=RefineIdsConfigs(bodyparts="--bpts_centre")
            ),
            extract_features=ExtractFeaturesConfigs(
                individuals="--indivs_simba", bodyparts="--bpts_simba"
            ),
            classify_behavs=[ClassifyBehavConfigs()],
            analyse=AnalyseConfigs(
                in_roi=[
                    InRoiConfigs(roi_corners="--bpts_corners", bodyparts="--bpts_front")
                ],
                speed=SpeedConfigs(bodyparts="--bpts_centre"),
                social_distance=SocialDistanceConfigs(bodyparts="--bpts_centre"),
                freezing=FreezingConfigs(bodyparts="--bpts_centre"),
            ),
        ),
        ref=RefConfigs.model_validate(
            {
                "indivs_simba": INDIVS_SIMBA,
                "bpts_simba": BPTS_SIMBA,
                "bpts_centre": BPTS_CENTRE,
                "bpts_front": BPTS_FRONT,
                "bpts_corners": BPTS_CORNERS,
            }
        ),
    )
