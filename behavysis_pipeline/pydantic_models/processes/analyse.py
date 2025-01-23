from behavysis_pipeline.constants import ARENA_BODYPARTS, SIMBA_BODYPARTS
from behavysis_pipeline.pydantic_models.pydantic_base_model import PydanticBaseModel


class SpeedConfigs(PydanticBaseModel):
    smoothing_sec: float | str = 1
    bodyparts: list[str] | str = SIMBA_BODYPARTS


class SocialDistanceConfigs(PydanticBaseModel):
    smoothing_sec: float | str = 1
    bodyparts: list[str] | str = SIMBA_BODYPARTS


class FreezingConfigs(PydanticBaseModel):
    window_sec: float | str = 2
    thresh_mm: float | str = 5
    smoothing_sec: float | str = 0.2
    bodyparts: list[str] | str = SIMBA_BODYPARTS


class InRoiConfigs(PydanticBaseModel):
    roi_name: str = "in_my_roi"
    is_in: bool | str = True
    thresh_mm: float | str = 0
    roi_corners: list[str] | str = ARENA_BODYPARTS
    bodyparts: list[str] | str = SIMBA_BODYPARTS


class AnalyseConfigs(PydanticBaseModel):
    bins_sec: list[int] | str = [30, 60, 120]
    custom_bins_sec: list[int] | str = [60, 120, 300, 600]

    speed: SpeedConfigs = SpeedConfigs()
    social_distance: SocialDistanceConfigs = SocialDistanceConfigs()
    freezing: FreezingConfigs = FreezingConfigs()
    in_roi: list[InRoiConfigs] = list()
