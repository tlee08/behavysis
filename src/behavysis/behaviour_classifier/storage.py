"""On-disk path functions for a self-contained behaviour classifier.

A classifier lives entirely inside its own directory (``clf_dir``). The
directory name is arbitrary — the behaviour it classifies is authored in each
model_type's ``config.yaml``, not inferred from the path. Training data lives
inside it too, mirroring the inference pipeline's stage folders::

    {clf_dir}/                       # arbitrary name (e.g. my_classifier/)
      training_data/                 # inference-pipeline files used for training
        5_features_extracted/
        7_behaviour_scored/
        ...
      production.yaml                # {model_type, version} currently deployed
      leaderboard.yaml               # auto-generated cross-model_type comparison
      {model_type}/                  # e.g. rf/, dnn1/, cnn2/
        config.yaml                  # human-authored recipe (holds behaviour_name)
        active.yaml                  # {version} pointer — best for this model_type
        versions/
          {version}/                 # e.g. v003_2025-07-07T120000
            model.joblib             # sklearn adapter (estimator + scaler)
            model.pt                 # torch state_dict
            scaler.joblib            # MinMaxScaler (torch only)
            metadata.yaml            # resolved hyperparams + eval summary
            dataset_manifest.yaml    # dataset hash + split experiment IDs
            evaluation/              # full eval artifacts (plots, reports)
"""

from __future__ import annotations

from pathlib import Path

from behavysis.constants import BEHAVIOUR_SCORED_DIR, FEATURES_EXTRACTED_DIR

TRAINING_DATA = "training_data"


# ── training data (inference-pipeline files) ─────────────────────────


def training_data_dir(clf_dir: Path) -> Path:
    """Directory holding the inference-pipeline files used for training."""
    return clf_dir / TRAINING_DATA


def features_dir(clf_dir: Path) -> Path:
    """Directory of extracted-feature parquet files for training."""
    return training_data_dir(clf_dir) / FEATURES_EXTRACTED_DIR


def labels_dir(clf_dir: Path) -> Path:
    """Directory of scored-behaviour parquet files for training."""
    return training_data_dir(clf_dir) / BEHAVIOUR_SCORED_DIR


# ── classifier level ─────────────────────────────────────────────────


def production_fp(clf_dir: Path) -> Path:
    """Path to the deployed-model pointer YAML."""
    return clf_dir / "production.yaml"


def leaderboard_fp(clf_dir: Path) -> Path:
    """Path to the cross-model_type leaderboard YAML."""
    return clf_dir / "leaderboard.yaml"


# ── model_type level ─────────────────────────────────────────────────


def model_type_dir(clf_dir: Path, model_type: str) -> Path:
    """Top-level directory for a single model_type."""
    return clf_dir / model_type


def config_fp(clf_dir: Path, model_type: str) -> Path:
    """Path to a model_type's human-authored training recipe YAML."""
    return model_type_dir(clf_dir, model_type) / "config.yaml"


def active_fp(clf_dir: Path, model_type: str) -> Path:
    """Path to a model_type's active-version pointer YAML."""
    return model_type_dir(clf_dir, model_type) / "active.yaml"


# ── version level ────────────────────────────────────────────────────


def versions_dir(clf_dir: Path, model_type: str) -> Path:
    """Directory containing all versions of a model_type."""
    return model_type_dir(clf_dir, model_type) / "versions"


def version_dir(clf_dir: Path, model_type: str, version: str) -> Path:
    """Directory for a single trained version's artifacts."""
    return versions_dir(clf_dir, model_type) / version


def metadata_fp(clf_dir: Path, model_type: str, version: str) -> Path:
    """Path to a version's metadata YAML."""
    return version_dir(clf_dir, model_type, version) / "metadata.yaml"


def dataset_manifest_fp(clf_dir: Path, model_type: str, version: str) -> Path:
    """Path to a version's dataset manifest YAML."""
    return version_dir(clf_dir, model_type, version) / "dataset_manifest.yaml"


def model_fp(clf_dir: Path, model_type: str, version: str) -> Path:
    """Return the version directory; the adapter decides which files to write."""
    return version_dir(clf_dir, model_type, version)


def eval_dir(clf_dir: Path, model_type: str, version: str) -> Path:
    """Directory for a version's evaluation artifacts (plots, reports)."""
    return version_dir(clf_dir, model_type, version) / "evaluation"
