import os

from behavysis.utils.pydantic_base_model import PydanticBaseModel


class ClassifyBehavConfigs(PydanticBaseModel):
    proj_dir: str = os.path.join("path", "to", "project_dir")
    behav_name: str = "behav_name"
    pcutoff: float | str = -1
    min_empty_window_secs: float | str = 0.2
    user_defined: list[str] | str = []
