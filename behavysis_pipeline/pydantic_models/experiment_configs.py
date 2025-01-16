"""
_summary_
"""

import os
from typing import Any

import matplotlib.pyplot as plt
from pydantic import BaseModel, ConfigDict, field_validator

from behavysis_pipeline.df_classes.keypoints_df import KeypointsDf
from behavysis_pipeline.pydantic_models.pydantic_base_model import PydanticBaseModel
from behavysis_pipeline.pydantic_models.vid_metadata import VidMetadata
from behavysis_pipeline.utils.misc_utils import enum2tuple

# TODO: have a function for selecting auto/user configs. May need to rejig ExperimentConfigs


class ConfigsFormatVid(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    width_px: None | int | str = None
    height_px: None | int | str = None
    fps: None | float | str = None
    start_sec: None | float | str = None
    stop_sec: None | float | str = None


class ConfigsRunDLC(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    model_fp: str = os.path.join(".")  # FilePath


class ConfigsCalculateParams(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="allow")


class ConfigsPreprocess(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="allow")


class ConfigsExtractFeatures(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    individuals: list[str] | str = ["mouse1marked", "mouse2unmarked"]
    bodyparts: list[str] | str = [
        "LeftEar",
        "RightEar",
        "Nose",
        "BodyCentre",
        "LeftFlankMid",
        "RightFlankMid",
        "TailBase1",
        "TailTip4",
    ]


class ConfigsClassifyBehav(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    model_fp: str = os.path.join(".")  # FilePath
    pcutoff: float | str = -1
    min_window_frames: int | str = 1
    user_behavs: list[str] | str = []


class ConfigsAnalyse(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="allow")

    bins_sec: list[int] | str = [30, 60, 120]
    custom_bins_sec: list[int] | str = [60, 120, 300, 600]


class ConfigsEvalKeypointsPlot(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    bodyparts: list[str] | str = [
        "LeftEar",
        "RightEar",
        "Nose",
        "BodyCentre",
        "LeftFlankMid",
        "RightFlankMid",
        "TailBase1",
        "TailTip4",
    ]


class ConfigsEvaluateVid(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    funcs: list[str] | str = ["keypoints", "analysis"]
    pcutoff: float | str = 0.8
    colour_level: str = KeypointsDf.CN.INDIVIDUALS.value
    radius: int | str = 3
    cmap: str = "rainbow"
    padding: int = 30

    @field_validator("cmap")
    @classmethod
    def validate_cmap(cls, v):
        """_summary_"""
        return PydanticBaseModel.validate_attr_closed_set(v, plt.colormaps())

    @field_validator("colour_level")
    @classmethod
    def validate_colour_level(cls, v):
        """_summary_"""
        vals = enum2tuple(KeypointsDf.CN)
        return PydanticBaseModel.validate_attr_closed_set(v, vals)


class ConfigsUser(BaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    format_vid: ConfigsFormatVid = ConfigsFormatVid()
    run_dlc: ConfigsRunDLC = ConfigsRunDLC()
    calculate_params: ConfigsCalculateParams = ConfigsCalculateParams()
    preprocess: ConfigsPreprocess = ConfigsPreprocess()
    extract_features: ConfigsExtractFeatures = ConfigsExtractFeatures()
    classify_behaviours: list[ConfigsClassifyBehav] = list()
    analyse: ConfigsAnalyse = ConfigsAnalyse()
    evaluate_vid: ConfigsEvaluateVid = ConfigsEvaluateVid()


class ConfigsAuto(PydanticBaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    raw_vid: VidMetadata = VidMetadata()
    formatted_vid: VidMetadata = VidMetadata()

    px_per_mm: float = -1
    start_frame: int = -1
    stop_frame: int = -1
    exp_dur_frames: int = -1
    scorer_name: str = "-1"


class ConfigsRef(PydanticBaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="allow")


class ExperimentConfigs(PydanticBaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    user: ConfigsUser = ConfigsUser()
    auto: ConfigsAuto = ConfigsAuto()
    ref: ConfigsRef = ConfigsRef()

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
            # Return the reference value
            return getattr(self.ref, val)
        # Return the value itself
        return val
