"""Behaviour classifier configuration model."""

from pathlib import Path

from pydantic import BaseModel, PositiveInt


class BehaviourClassifierConfig(BaseModel):
    """Behaviour Classifier Config Model."""

    proj_dir: Path = Path("project_dir")
    behaviour_name: str = "behaviour_name"
    model_type: str = "rf"
    seed: int = 42
    oversample_ratio: float = 0.2
    undersample_ratio: float = 0.4

    feature_start_col: int = 48
    nfeatures: int | None = None
    window_frames: int = 0

    pcutoff: float = 0.2
    test_split: float = 0.2
    val_split: float = 0.2
    batch_size: PositiveInt = 256
    epochs: PositiveInt = 100
