from matplotlib import pyplot as plt
from pydantic import field_validator

from behavysis_pipeline.df_classes.keypoints_df import KeypointsDf
from behavysis_pipeline.pydantic_models.pydantic_base_model import PydanticBaseModel
from behavysis_pipeline.utils.misc_utils import enum2tuple


class EvaluateVidConfigs(PydanticBaseModel):
    funcs: list[str] | str = ["keypoints", "analysis"]
    pcutoff: float | str = 0.8
    colour_level: str = KeypointsDf.CN.INDIVIDUALS.value
    radius: int | str = 3
    cmap: str = "rainbow"
    padding: int = 30

    @field_validator("cmap")
    @classmethod
    def validate_cmap(cls, v):
        return cls.validate_attr_closed_set(v, plt.colormaps())

    @field_validator("colour_level")
    @classmethod
    def validate_colour_level(cls, v):
        vals = enum2tuple(KeypointsDf.CN)
        return cls.validate_attr_closed_set(v, vals)


class KeypointsConfigs(PydanticBaseModel):
    pcutoff: float | str = 0.8
    colour_level: str = KeypointsDf.CN.INDIVIDUALS.value
    radius: int | str = 3
    cmap: str = "rainbow"

    @field_validator("cmap")
    @classmethod
    def validate_cmap(cls, v):
        return cls.validate_attr_closed_set(v, plt.colormaps())

    @field_validator("colour_level")
    @classmethod
    def validate_colour_level(cls, v):
        vals = enum2tuple(KeypointsDf.CN)
        return cls.validate_attr_closed_set(v, vals)


class AnalysisConfigs(PydanticBaseModel):
    padding: int = 30  # Graph viewable window either side


class JohanssonConfigs(PydanticBaseModel):
    colour: str = "black"
