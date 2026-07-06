"""Behaviour classifier configuration model.

Serialised as ``config.yaml`` alongside trained models.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, PositiveInt


class BehaviourClassifierConfig(BaseModel):
    """Configuration for a behaviour classifier.

    Defines model identity (individuals, bodyparts, active model type)
    and training/inference hyperparameters.
    """

    behaviour_name: str
    model_type: str = "rf"
    individuals: list[str] = []
    bodyparts: list[str] = []
    pcutoff: float = 0.2
    seed: int = 42
    oversample_ratio: float = 0.2
    undersample_ratio: float = 0.4
    test_split: float = 0.2
    val_split: float = 0.2
    batch_size: PositiveInt = 256
    epochs: PositiveInt = 100

    @classmethod
    def read_yaml(cls, fp: Path) -> "BehaviourClassifierConfig":
        """Read config from a YAML file."""
        return cls.model_validate(yaml.safe_load(fp.open("r")))

    def write_yaml(self, fp: Path) -> None:
        """Write config to a YAML file."""
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(yaml.dump(self.model_dump(), default_flow_style=False))
