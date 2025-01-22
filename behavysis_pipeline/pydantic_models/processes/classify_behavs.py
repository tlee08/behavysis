import os

from behavysis_pipeline.pydantic_models.pydantic_base_model import PydanticBaseModel


class ClassifyBehavConfigs(PydanticBaseModel):
    proj_dir: str = os.path.join(".")
    behav_name: str = "behav_name"
    pcutoff: float | str = -1
    min_window_frames: int | str = 1
    user_defined: list[str] | str = []
