from typing import Literal

from pydantic import BaseModel

from behavysis.constants import BPTS_SIMBA


class InterpolateConfigs(BaseModel):
    pcutoff: float | str = 0.5


class InterpolateStationaryConfigs(BaseModel):
    bodypart: str = "bodypart"
    pcutoff: float = 0.8
    pcutoff_all: float = 0.6
    x: float = 0
    y: float = 0


class RefineIdsConfigs(BaseModel):
    marked: str = "marked"
    unmarked: str = "unmarked"
    marking: str = "marking"
    bodyparts: list[str] | str = BPTS_SIMBA
    window_sec: float | str = 0.5
    metric: Literal["current", "rolling", "binned"] | str = "current"


class PreprocessConfigs(BaseModel):
    interpolate: InterpolateConfigs = InterpolateConfigs()
    interpolate_stationary: list[InterpolateStationaryConfigs] = list()
    refine_ids: RefineIdsConfigs = RefineIdsConfigs()
