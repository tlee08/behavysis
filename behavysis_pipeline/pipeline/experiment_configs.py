"""
_summary_
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
from pydantic import BaseModel, ConfigDict, FilePath, field_validator

from behavysis_core.data_models.pydantic_base_model import PydanticBaseModel
from behavysis_core.data_models.vid_metadata import VidMetadata
from behavysis_core.utils.constants import DLC_COLUMN_NAMES


class ConfigsUnitProcess(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="allow")


class ConfigsFormatVid(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    width_px: Optional[int] = None
    height_px: Optional[int] = None
    fps: Optional[int] = None
    start_sec: Optional[float] = None
    stop_sec: Optional[float] = None


class ConfigsRunDLC(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    dlc_config_path: FilePath = os.path.join(".")  # type: ignore


class ConfigsCalculateParams(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    start_frame: ConfigsUnitProcess = ConfigsUnitProcess()
    stop_frame: ConfigsUnitProcess = ConfigsUnitProcess()
    px_per_mm: ConfigsUnitProcess = ConfigsUnitProcess()


class ConfigsPreprocess(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    interpolate_points: ConfigsUnitProcess = ConfigsUnitProcess()
    bodycentre: ConfigsUnitProcess = ConfigsUnitProcess()
    refine_identities: ConfigsUnitProcess = ConfigsUnitProcess()


class ConfigsExtractFeatures(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    individuals: list[str] = ["mouse1marked", "mouse2unmarked"]
    bodyparts: list[str] = [
        "LeftEar",
        "RightEar",
        "Nose",
        "BodyCentre",
        "LeftFlankMid",
        "RightFlankMid",
        "TailBase1",
        "TailTip4",
    ]


class ConfigsClassifyBehaviours(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    models: list[FilePath] = []
    min_window_frames: int = 1
    user_behavs: list[str] = []


class ConfigsAnalyse(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    bins_sec: list[int] = [30, 60, 120]
    custom_bins_sec: list[int] = [60, 120, 300, 600]

    thigmotaxis: ConfigsUnitProcess = ConfigsUnitProcess()
    center_crossing: ConfigsUnitProcess = ConfigsUnitProcess()
    speed: ConfigsUnitProcess = ConfigsUnitProcess()
    social_distance: ConfigsUnitProcess = ConfigsUnitProcess()
    freezing: ConfigsUnitProcess = ConfigsUnitProcess()


class ConfigsEvalKeypointsPlot(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    bodyparts: list[str] = [
        "LeftEar",
        "RightEar",
        "Nose",
        "BodyCentre",
        "LeftFlankMid",
        "RightFlankMid",
        "TailBase1",
        "TailTip4",
    ]


class ConfigsEvalVid(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    funcs: list[str] = ["keypoints"]
    pcutoff: float = 0.8
    colour_level: str = "individuals"
    radius: int = 3
    cmap: str = "rainbow"

    @field_validator("cmap")
    @classmethod
    def validate_cmap(cls, v):
        """_summary_"""
        return PydanticBaseModel.validate_attr_closed_set(v, plt.colormaps())

    @field_validator("colour_level")
    @classmethod
    def validate_colour_level(cls, v):
        """_summary_"""
        return PydanticBaseModel.validate_attr_closed_set(v, DLC_COLUMN_NAMES)


class ConfigsEvaluate(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    keypoints_plot: ConfigsEvalKeypointsPlot = ConfigsEvalKeypointsPlot()
    eval_vid: ConfigsEvalVid = ConfigsEvalVid()


class ConfigsUser(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    format_vid: ConfigsFormatVid = ConfigsFormatVid()
    run_dlc: ConfigsRunDLC = ConfigsRunDLC()
    calculate_params: ConfigsCalculateParams = ConfigsCalculateParams()
    preprocess: ConfigsPreprocess = ConfigsPreprocess()
    extract_features: ConfigsExtractFeatures = ConfigsExtractFeatures()
    classify_behaviours: ConfigsClassifyBehaviours = ConfigsClassifyBehaviours()
    analyse: ConfigsAnalyse = ConfigsAnalyse()
    evaluate: ConfigsEvaluate = ConfigsEvaluate()


class ConfigsAuto(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    raw_vid: VidMetadata = VidMetadata()
    formatted_vid: VidMetadata = VidMetadata()
    px_per_mm: Optional[float] = None
    start_frame: Optional[int] = None
    stop_frame: Optional[int] = None


class ExperimentConfigs(PydanticBaseModel, BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    user: ConfigsUser = ConfigsUser()
    auto: ConfigsAuto = ConfigsAuto()
