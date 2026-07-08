"""Training data snapshot — populates a classifier's training_data/ directory.

Mirrors the behavysis project pipeline folder structure so the behaviour
viewer and ``train()`` can consume it directly::

    training_data/
      0_config/                 (symlinks → proj/0_config/*)
      0_metadata/               (symlinks → proj/0_metadata/*)
      2_formatted_videos/       (symlinks → proj/2_formatted_videos/*)
      3_keypoints/              (symlinks → proj/3_keypoints/*)
      5_features_extracted/     (copies from proj/5_features_extracted/*)
      7_behaviour_scored/       (copies from proj/7_behaviour_scored/*)
"""

import shutil
from pathlib import Path

from loguru import logger

from behavysis.constants.pipeline import (
    FEATURES_EXTRACTED_DIR,
    KEYPOINTS_DIR,
    BEHAVIOUR_SCORED_DIR,
    CONFIG_DIR,
    FORMATTED_VIDEO_DIR,
    METADATA_DIR,
)


_SYMLINK_STAGES = [
    CONFIG_DIR,
    METADATA_DIR,
    FORMATTED_VIDEO_DIR,
    KEYPOINTS_DIR,
]

_COPY_STAGES = [
    FEATURES_EXTRACTED_DIR,
    BEHAVIOUR_SCORED_DIR,
]


def populate_training_data(
    clf_dir: Path,
    proj_dir: Path,
    experiment_names: list[str],
) -> None:
    """Populate ``clf_dir/training_data/`` from experiment files in ``proj_dir``.

    Configs, metadata, videos, and keypoints are symlinked (read-only).
    Features and scored behaviour are copied so they're self-contained.
    """
    td = clf_dir / "training_data"
    td.mkdir(parents=True, exist_ok=True)

    for stage in _SYMLINK_STAGES:
        _symlink_stage(proj_dir, td, stage, experiment_names)

    for stage in _COPY_STAGES:
        _copy_stage(proj_dir, td, stage, experiment_names)

    logger.info(
        "Populated training_data/ in %s with %d experiment(s)",
        clf_dir,
        len(experiment_names),
    )


def _symlink_stage(
    proj_dir: Path,
    td: Path,
    stage: str,
    names: list[str],
) -> None:
    """Symlink experiment files from proj_dir/stage/ into td/stage/."""
    src_dir = proj_dir / stage
    stage_dir = td / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        for src in src_dir.glob(f"{name}.*"):
            dst = stage_dir / src.name
            if not dst.exists():
                dst.symlink_to(src)


def _copy_stage(
    proj_dir: Path,
    td: Path,
    stage: str,
    names: list[str],
) -> None:
    """Copy experiment files from proj_dir/stage/ into td/stage/."""
    src_dir = proj_dir / stage
    stage_dir = td / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        for src in src_dir.glob(f"{name}.*"):
            dst = stage_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
