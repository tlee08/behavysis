from pathlib import Path

from pydantic import BaseModel


class RunDlcConfigs(BaseModel):
    model_fp: Path = Path("path") / "to" / "DEEPLABCUT_model" / "config.yaml"
