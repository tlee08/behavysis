from typing import Literal

from behavysis_pipeline.pydantic_models.pydantic_base_model import PydanticBaseModel


class InterpolateConfigs(PydanticBaseModel):
    pcutoff: float | str = 0


class InterpolateStationaryConfigs(PydanticBaseModel):
    bodypart: str = ""
    pcutoff: float = 0.8
    pcutoff_all: float = 0.6
    x: float = 0
    y: float = 0


class RefineIdsConfigs(PydanticBaseModel):
    marked: str = ""
    unmarked: str = ""
    marking: str = ""
    bodyparts: list[str] | str = []
    window_sec: float | str = 0
    metric: Literal["current", "rolling", "binned"] | str = "current"


class PreprocessConfigs(PydanticBaseModel):
    interpolate: InterpolateConfigs = InterpolateConfigs()
    interpolate_stationary: InterpolateStationaryConfigs = InterpolateStationaryConfigs()
    refine_ids: RefineIdsConfigs = RefineIdsConfigs()
