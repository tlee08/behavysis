"""
_summary_
"""

from pydantic import ConfigDict

from behavysis_pipeline.pydantic_models.pydantic_base_model import PydanticBaseModel


class VidMetadata(PydanticBaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    fps: float = -1
    width_px: int = -1
    height_px: int = -1
    total_frames: int = -1
