"""
_summary_
"""

from typing import Any

from pydantic import ConfigDict

from behavysis_pipeline.pydantic_models.processes.analyse import AnalyseConfigs
from behavysis_pipeline.pydantic_models.processes.calculate_params import CalculateParamsConfigs
from behavysis_pipeline.pydantic_models.processes.classify_behavs import ClassifyBehavConfigs
from behavysis_pipeline.pydantic_models.processes.evaluate_vid import EvaluateVidConfigs
from behavysis_pipeline.pydantic_models.processes.extract_features import ExtractFeaturesConfigs
from behavysis_pipeline.pydantic_models.processes.format_vid import FormatVidConfigs, VidMetadata
from behavysis_pipeline.pydantic_models.processes.preprocess import PreprocessConfigs
from behavysis_pipeline.pydantic_models.processes.run_dlc import RunDlcConfigs
from behavysis_pipeline.pydantic_models.pydantic_base_model import PydanticBaseModel

# class EvalKeypointsPlotConfigs(PydanticBaseModel):
#     bodyparts: list[str] | str = SIMBA_BODYPARTS


class UserConfigs(PydanticBaseModel):
    format_vid: FormatVidConfigs = FormatVidConfigs()
    run_dlc: RunDlcConfigs = RunDlcConfigs()
    calculate_params: CalculateParamsConfigs = CalculateParamsConfigs()
    preprocess: PreprocessConfigs = PreprocessConfigs()
    extract_features: ExtractFeaturesConfigs = ExtractFeaturesConfigs()
    classify_behavs: list[ClassifyBehavConfigs] = list()
    analyse: AnalyseConfigs = AnalyseConfigs()
    evaluate_vid: EvaluateVidConfigs = EvaluateVidConfigs()


class AutoConfigs(PydanticBaseModel):
    raw_vid: VidMetadata = VidMetadata()
    formatted_vid: VidMetadata = VidMetadata()

    px_per_mm: float = -1
    start_frame: int = -1
    stop_frame: int = -1
    dur_frames: int = -1
    scorer_name: str = "scorer"


class RefConfigs(PydanticBaseModel):
    model_config = ConfigDict(extra="allow")


class ExperimentConfigs(PydanticBaseModel):
    user: UserConfigs = UserConfigs()
    auto: AutoConfigs = AutoConfigs()
    ref: RefConfigs = RefConfigs()

    def get_ref(self, val: Any) -> Any:
        """
        If the val is in the reference format, then
        return reference value of the val if it exists in the reference store.
        Otherwise, return the val itself.

        Note
        ----
        The reference format is `"--<ref_name>"`.
        """
        # Check if the value is in the reference format
        if isinstance(val, str) and val.startswith("--"):
            # Remove the '--' from the val
            val = val[2:]
            # Check if the value exists in the reference store
            assert hasattr(self.ref, val), f"Value '{val}' can't be found in the configs reference section."
            return getattr(self.ref, val)
        return val

    def get_analysis_configs(self) -> tuple[float, float, float, float, list, list]:
        """
        _summary_

        Parameters
        ----------
        configs : Configs
            _description_

        Returns
        -------
        tuple[ float, float, float, float, list, list, ]
            _description_
        """
        assert self.auto.formatted_vid.fps
        assert self.auto.formatted_vid.width_px
        assert self.auto.formatted_vid.height_px
        assert self.auto.px_per_mm
        return (
            float(self.auto.formatted_vid.fps),
            float(self.auto.formatted_vid.width_px),
            float(self.auto.formatted_vid.height_px),
            float(self.auto.px_per_mm),
            list(self.get_ref(self.user.analyse.bins_sec)),
            list(self.get_ref(self.user.analyse.custom_bins_sec)),
        )
