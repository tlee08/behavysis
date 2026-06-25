from pathlib import Path

from pydantic import BaseModel


class RunDlcConfig(BaseModel):
    """RunDlcConfig."""

    model_fp: Path = Path("path") / "to" / "DEEPLABCUT_model" / "config.yaml"
