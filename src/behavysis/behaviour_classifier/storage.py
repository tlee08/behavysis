"""On-disk layout and paths for classifier artifacts."""

from pathlib import Path

MODEL_ROOT = "behaviour_models"


def model_dir(proj_dir: Path, behaviour_name: str) -> Path:
    return proj_dir / MODEL_ROOT / behaviour_name


def config_fp(proj_dir: Path, behaviour_name: str) -> Path:
    return model_dir(proj_dir, behaviour_name) / "config.json"


def classifier_fp(proj_dir: Path, behaviour_name: str, model_type: str) -> Path:
    return model_dir(proj_dir, behaviour_name) / model_type / "classifier.sav"


def eval_dir(proj_dir: Path, behaviour_name: str, model_type: str) -> Path:
    return model_dir(proj_dir, behaviour_name) / model_type / "evaluation"


def training_data_dir(proj_dir: Path, behaviour_name: str, model_type: str) -> Path:
    return model_dir(proj_dir, behaviour_name) / model_type / "training_data"
