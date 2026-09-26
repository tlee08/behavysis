"""YAML model base class for Pydantic models."""

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict


class YamlModel(BaseModel):
    """Base model with YAML read/write helpers."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def read_yaml(cls, fp: Path) -> Self:
        """Read the model from a YAML file, dropping ``_`` anchor keys."""
        data = yaml.safe_load(fp.read_text())
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if not k.startswith("_")}
        return cls.model_validate(data)

    def write_yaml(self, fp: Path) -> None:
        """Write the model to a YAML file."""
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(yaml.dump(self.model_dump(mode="json"), default_flow_style=False))
