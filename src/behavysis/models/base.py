"""YAML model base class for Pydantic models."""

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel


class YamlModel(BaseModel):
    """Base model with YAML read/write helpers."""

    @classmethod
    def read_yaml(cls, fp: Path) -> Self:
        """Read the model from a YAML file."""
        return cls.model_validate(yaml.safe_load(fp.read_text()))

    def write_yaml(self, fp: Path) -> None:
        """Write the model to a YAML file."""
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(yaml.dump(self.model_dump(mode="json"), default_flow_style=False))
