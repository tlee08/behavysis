from pydantic import BaseModel

from behavysis.constants import BPTS_CORNERS, BPTS_SIMBA


class SpeedConfig(BaseModel):
    """SpeedConfig."""

    smoothing_sec: float | str = 1
    bodyparts: list[str] | str = BPTS_SIMBA


class SocialDistanceConfig(BaseModel):
    """SocialDistanceConfig."""

    smoothing_sec: float | str = 1
    bodyparts: list[str] | str = BPTS_SIMBA


class FreezingConfig(BaseModel):
    """FreezingConfig."""

    window_sec: float | str = 2.0
    thresh_mm: float | str = 5.0
    smoothing_sec: float | str = 0.2
    bodyparts: list[str] | str = BPTS_SIMBA


class InRoiConfig(BaseModel):
    """InRoiConfig."""

    roi_name: str = "in_my_roi"
    is_in: bool | str = True
    padding_mm: float | str = 0.0
    roi_corners: list[str] | str = BPTS_CORNERS
    bodyparts: list[str] | str = BPTS_SIMBA


class AnalyseConfig(BaseModel):
    """AnalyseConfig."""

    bins_sec: list[int] | str = [30, 60, 120]
    custom_bins_sec: list[int] | str = [60, 120, 300, 600]

    speed: SpeedConfig = SpeedConfig()
    social_distance: SocialDistanceConfig = SocialDistanceConfig()
    freezing: FreezingConfig = FreezingConfig()
    in_roi: list[InRoiConfig] = []
