"""On-disk layout and paths for classifier artifacts.

Directory structure::

    behaviour_models/
      {behaviour}/
        config.yaml              # BehaviourClassifierConfig
        {model_type}/
          model.sav              # joblib serialised adapter
          evaluation/            # eval plots and reports
          training_data/         # training snapshot
"""

from pathlib import Path

MODEL_ROOT = "behaviour_models"


def model_dir(proj_dir: Path, behaviour_name: str) -> Path:
    """Top-level directory for a behaviour's models and config."""
    return proj_dir / MODEL_ROOT / behaviour_name


def config_fp(proj_dir: Path, behaviour_name: str) -> Path:
    """Path to the classifier config YAML file."""
    return model_dir(proj_dir, behaviour_name) / "config.yaml"


def classifier_fp(proj_dir: Path, behaviour_name: str, model_type: str) -> Path:
    """Path to the serialised model adapter."""
    return model_dir(proj_dir, behaviour_name) / model_type / "model.sav"


def eval_dir(proj_dir: Path, behaviour_name: str, model_type: str) -> Path:
    """Directory for evaluation artifacts."""
    return model_dir(proj_dir, behaviour_name) / model_type / "evaluation"


def training_data_dir(proj_dir: Path, behaviour_name: str, model_type: str) -> Path:
    """Directory for training data snapshots."""
    return model_dir(proj_dir, behaviour_name) / model_type / "training_data"
