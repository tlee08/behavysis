"""
_summary_
"""

from typing import Optional, Union

import matplotlib.pyplot as plt
from pydantic import (  # FilePath,
    BaseModel,
    ValidationError,
    field_validator,
    model_validator,
)

from ba_pipeline.utils.constants import DLC_COLUMN_NAMES


def _validate_attrs_as_unit_processes(model, field_names):
    for k in field_names:
        try:
            v = getattr(model, k)
            setattr(model, k, ConfigsUnitProcess.model_validate(v))
        except Exception as e:
            raise ValidationError(f"'{k}' is not a dict\n:" + f"{k}: {v}") from e
    return model


def _check_attrs_in_set(v, valid_options):
    if v not in valid_options:
        raise ValidationError(f"Option must be one of: {', '.join(valid_options)}")
    return v


class ConfigsUnitProcess(BaseModel):
    """_summary_"""

    class Config:
        """_summary_"""

        extra = "allow"


class ConfigsVidMetadata(BaseModel):
    """_summary_"""

    fps: Optional[Union[int, float]] = None
    width_px: Optional[int] = None
    height_px: Optional[int] = None
    total_frames: Optional[int] = None

    class Config:
        """_summary_"""

        extra = "allow"


class ConfigsFormatVid(BaseModel):
    """_summary_"""

    width_px: Optional[int] = None
    height_px: Optional[int] = None
    fps: Optional[int] = None
    start_sec: Optional[float] = None
    stop_sec: Optional[float] = None


class ConfigsRunDLC(BaseModel):
    """_summary_"""

    dlc_config_path: str = "."  # FilePath


class ConfigsCalculateParams(BaseModel):
    """_summary_"""

    class Config:
        """_summary_"""

        extra = "allow"

    @model_validator(mode="after")
    @classmethod
    def check_attrs_are_types(cls, model):
        """_summary_"""
        return _validate_attrs_as_unit_processes(model, model.model_extra.keys())


class ConfigsPreprocess(BaseModel):
    """_summary_"""

    class Config:
        """_summary_"""

        extra = "allow"

    @model_validator(mode="after")
    @classmethod
    def check_attrs_are_types(cls, model):
        """_summary_"""
        return _validate_attrs_as_unit_processes(model, model.model_extra.keys())


class ConfigsExtractFeatures(BaseModel):
    """_summary_"""

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

    models: list[str] = ["."]  # FilePath
    min_window_frames: int = 1
    user_behavs: list[str] = ["example"]


class ConfigsAnalyse(BaseModel):
    """_summary_"""

    bins_sec: list[int] = [30, 60, 120]
    custom_bins_sec: list[int] = [60, 120, 300, 600]

    class Config:
        """_summary_"""

        extra = "allow"

    @model_validator(mode="after")
    @classmethod
    def check_attrs_are_types(cls, model):
        """_summary_"""
        return _validate_attrs_as_unit_processes(model, list(model.model_extra.keys()))


class ConfigsEvalKeypointsPlot(BaseModel):
    """_summary_"""

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

    funcs: list[str] = ["keypoints"]
    pcutoff: float = 0.8
    colour_level: str = "individuals"
    radius: int = 3
    cmap: str = "rainbow"

    @field_validator("cmap")
    @classmethod
    def check_cmap(cls, v):
        """_summary_"""
        return _check_attrs_in_set(v, plt.colormaps())

    @field_validator("colour_level")
    @classmethod
    def check_colour_level(cls, v):
        """_summary_"""
        return _check_attrs_in_set(v, DLC_COLUMN_NAMES)


class ConfigsEvaluate(BaseModel):
    """_summary_"""

    keypoints_plot: ConfigsEvalKeypointsPlot = ConfigsEvalKeypointsPlot()
    eval_vid: ConfigsEvalVid = ConfigsEvalVid()


class ConfigsUser(BaseModel):
    """_summary_"""

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

    raw_vid: ConfigsVidMetadata = ConfigsVidMetadata()
    formatted_vid: ConfigsVidMetadata = ConfigsVidMetadata()
    px_per_mm: Optional[float] = None
    start_frame: Optional[int] = None
    stop_frame: Optional[int] = None


class ExperimentConfigs(BaseModel):
    """_summary_"""

    user: ConfigsUser = ConfigsUser()
    auto: ConfigsAuto = ConfigsAuto()
