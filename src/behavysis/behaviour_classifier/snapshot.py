"""Training data snapshot for reproducibility and behavioral validation."""

import json
from datetime import UTC, datetime

import numpy as np
import polars as pl

from .config import BehaviourClassifierConfig
from .storage import training_data_dir


class TrainingSnapshot:
    """Saves training data as parquet + video symlinks for reproducibility."""

    @classmethod
    def create(
        cls,
        x_ls: list[np.ndarray],
        y_ls: list[np.ndarray],
        experiment_names: list[str],
        config: BehaviourClassifierConfig,
    ) -> None:
        snap_dir = training_data_dir(
            config.proj_dir,
            config.behaviour_name,
            config.model_type,
        )

        # Features
        feat_dir = snap_dir / "features"
        feat_dir.mkdir(parents=True, exist_ok=True)
        for name, x in zip(experiment_names, x_ls, strict=True):
            pl.DataFrame(
                x,
                schema=[f"f{i}" for i in range(x.shape[1])],
            ).write_parquet(feat_dir / f"{name}.parquet")

        # Labels
        label_dir = snap_dir / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        for name, y in zip(experiment_names, y_ls, strict=True):
            pl.DataFrame(
                y.reshape(-1, 1),
                schema=[config.behaviour_name],
            ).write_parquet(label_dir / f"{name}.parquet")

        # Video symlinks
        vid_dir = snap_dir / "videos"
        vid_dir.mkdir(parents=True, exist_ok=True)
        for name in experiment_names:
            src = config.proj_dir / "2_formatted_videos" / f"{name}.mp4"
            dst = vid_dir / f"{name}.mp4"
            if src.exists() and not dst.exists():
                dst.symlink_to(src)

        # Manifest
        (snap_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "behaviour_name": config.behaviour_name,
                    "model_type": config.model_type,
                    "experiments": experiment_names,
                    "n_experiments": len(experiment_names),
                    "n_samples": sum(x.shape[0] for x in x_ls),
                    "n_features": x_ls[0].shape[1],
                    "created_at": datetime.now(tz=UTC).isoformat(),
                },
                indent=2,
            ),
        )
