from pathlib import Path

from pydantic import BaseModel


class ClassifyBehavConfigs(BaseModel):
    proj_dir: Path = Path("path") / "to" / "project_dir"
    behav_name: str = "behav_name"
    pcutoff: float | str = -1
    min_empty_window_secs: float | str = 0.2
    user_defined: list[str] | str = []
