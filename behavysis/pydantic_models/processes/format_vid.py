from behavysis.utils.pydantic_base_model import PydanticBaseModel


class FormatVidConfigs(PydanticBaseModel):
    width_px: None | int | str = None
    height_px: None | int | str = None
    fps: None | float | str = None
    start_sec: None | float | str = None
    stop_sec: None | float | str = None


class VidMetadata(PydanticBaseModel):
    width_px: int = -1
    height_px: int = -1
    fps: float = -1
    total_frames: int = -1
