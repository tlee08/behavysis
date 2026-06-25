from typing import Literal

from pydantic import BaseModel

from behavysis.constants import BPTS_SIMBA


class InterpolateConfig(BaseModel):
    """InterpolateConfig."""

    pcutoff: float | str = 0.5


class InterpolateStationaryConfig(BaseModel):
    """InterpolateStationaryConfig."""

    bodypart: str = "bodypart"
    pcutoff: float = 0.8
    pcutoff_all: float = 0.6
    x: float = 0
    y: float = 0


class RefineIdsConfig(BaseModel):
    """RefineIdsConfig."""

    marked: str = "marked"
    unmarked: str = "unmarked"
    marking: str = "marking"
    bodyparts: list[str] | str = BPTS_SIMBA
    window_sec: float | str = 0.5
    metric: Literal["current", "rolling", "binned"] = "current"


class PreprocessConfig(BaseModel):
    """PreprocessConfig."""

    interpolate: InterpolateConfig = InterpolateConfig()
    interpolate_stationary: list[InterpolateStationaryConfig] = []
    refine_ids: RefineIdsConfig = RefineIdsConfig()
