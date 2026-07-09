"""Experiment class for processing a single experiment in the behavysis pipeline."""

import shutil
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from behavysis.behaviour_classifier import ProductionPointer
from behavysis.constants import (
    ANALYSIS_COMBINED_DIR,
    ANALYSIS_DIR,
    CONFIG_DIR,
    FORMATTED_VIDEO_DIR,
    KEYPOINTS_DIR,
    METADATA_DIR,
    RAW_VIDEO_DIR,
    STAGES,
)
from behavysis.constants.pipeline import (
    BEHAVIOUR_PREDICTED_DIR,
    BEHAVIOUR_SCORED_DIR,
    FEATURES_EXTRACTED_DIR,
    PREPROCESSED_DIR,
)
from behavysis.funcs import (
    AnalyseFunc,
    CalculateParametersFunc,
    PreprocessFunc,
    analyse_behaviour,
    classify_behaviour,
    combine_analysis,
    extract_features,
    format_video,
    get_vid_metadata,
    ma_dlc_run_single,
)
from behavysis.models import BoutStruct, ExperimentConfig, ExperimentMetadata
from behavysis.schemas import (
    BEHAVIOUR_PREDICTED_SCHEMA,
    BEHAVIOUR_SCORED_BASE,
    KEYPOINTS_SCHEMA,
    read_df,
    write_df,
)
from behavysis.transforms import predicted_to_scored
from behavysis.utils import log_file_exists, trace


def _get_frame(vid_fp: Path, metadata: ExperimentMetadata) -> np.ndarray:
    """Extract frame 150 (0-indexed 149) for background plots, or black frame."""
    cap = cv2.VideoCapture(str(vid_fp))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 149)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return np.zeros(
            (metadata.require_height_px(), metadata.require_width_px(), 3),
            dtype=np.uint8,
        )
    return frame


class Experiment:
    """Behavysis Pipeline class for a single experiment."""

    name: str
    root_dir: Path

    def __init__(self, name: str, root_dir: str | Path) -> None:
        """Initialises the experiment with the given name and root directory."""
        self.name = name
        self.root_dir = Path(root_dir)
        # Check root_dir exists
        if not self.root_dir.is_dir():
            msg = (
                f'Project folder not found: "{root_dir}"\n'
                f"  Create a new project with: behavysis-make-project"
            )
            raise ValueError(msg)
        # Check experiment name exists in root_dir
        if not np.any([self.get_fp(f).is_file() for f in STAGES]):
            folders_ls_msg = "".join([f"\n    - {f}" for f in STAGES])
            msg = (
                f'No files named "{name}" found in "{root_dir}".\n'
                f"  Expected files in one of these folders:{folders_ls_msg}\n"
                "  Tip: Check the experiment name matches your file names "
                "(without extension)."
            )
            raise ValueError(msg)
        # Set initial metadata
        metadata = self.read_metadata()
        metadata.name = self.name
        self.write_metadata(metadata)

    def get_log_context(self) -> dict:
        """Log context for loguru context."""
        return {"experiment": str(self.name)}

    def get_fp(self, folder: str) -> Path:
        """Returns the experiment's file path from the given folder."""
        return self.root_dir / folder / f"{self.name}.{STAGES[folder]}"

    def read_config(self) -> ExperimentConfig:
        """Returns the experiment's config."""
        return ExperimentConfig.read_yaml(self.get_fp(CONFIG_DIR))

    def read_metadata(self) -> ExperimentMetadata:
        """Returns the experiment's metadata."""
        if not self.get_fp(METADATA_DIR).exists():
            return ExperimentMetadata()
        return ExperimentMetadata.model_validate_json(
            self.get_fp(METADATA_DIR).read_text(),
        )

    def write_metadata(self, metadata: ExperimentMetadata) -> None:
        """Save the experiment's metadata to disk."""
        self.get_fp(METADATA_DIR).write_text(metadata.model_dump_json(indent=2))

    @trace
    def update_config(
        self,
        default_config_fp: Path,
    ) -> None:
        """Initialises the JSON config files with the given configurations."""
        # Parsing in the new config to see if it is valid
        ExperimentConfig.read_yaml(default_config_fp)
        # Overwriting the config file with the new config
        shutil.copyfile(default_config_fp, self.get_fp(CONFIG_DIR))

    @trace
    def format_video(self, *, overwrite: bool) -> None:
        """Formats the video with ffmpeg to fit the formatted config."""
        # Overwrite check
        if not overwrite and self.get_fp(FORMATTED_VIDEO_DIR).exists():
            log_file_exists(self.get_fp(FORMATTED_VIDEO_DIR))
            return
        # Process
        format_video(
            raw_vid_fp=self.get_fp(RAW_VIDEO_DIR),
            formatted_vid_fp=self.get_fp(FORMATTED_VIDEO_DIR),
            config=self.read_config(),
        )
        # Update metadata
        self.get_vid_metadata()

    @trace
    def get_vid_metadata(self) -> None:
        """Get vid metadata and save."""
        # Read
        metadata = self.read_metadata()
        # Update
        metadata.raw_video = get_vid_metadata(self.get_fp(RAW_VIDEO_DIR))
        metadata.formatted_video = get_vid_metadata(self.get_fp(FORMATTED_VIDEO_DIR))
        # Write
        self.write_metadata(metadata)

    @trace
    def run_dlc(self, gputouse: int | None, *, overwrite: bool) -> None:
        """Run the DLC model on the formatted video."""
        # Overwrite check
        if not overwrite and self.get_fp(KEYPOINTS_DIR).exists():
            log_file_exists(self.get_fp(KEYPOINTS_DIR))
            return
        # Process
        ma_dlc_run_single(
            vid_fp=self.get_fp(FORMATTED_VIDEO_DIR),
            keypoints_dir=self.root_dir / KEYPOINTS_DIR,
            config=self.read_config(),
            gputouse=gputouse,
        )

    @trace
    def calculate_parameters(self, funcs: tuple[CalculateParametersFunc, ...]) -> None:
        """Calculate parameters of the keypoints file."""
        keypoints_df = read_df(self.get_fp(KEYPOINTS_DIR), KEYPOINTS_SCHEMA)
        metadata = self.read_metadata()
        for func in funcs:
            metadata = func(
                keypoints_df=keypoints_df,
                config=self.read_config(),
                metadata=metadata,
            )
        # Write
        self.write_metadata(metadata)

    @trace
    def preprocess(self, funcs: tuple[PreprocessFunc, ...], *, overwrite: bool) -> None:
        """Preprocessing pipeline for keypoints data."""
        # Overwrite check
        if not overwrite and self.get_fp(PREPROCESSED_DIR).exists():
            log_file_exists(self.get_fp(PREPROCESSED_DIR))
            return
        # Process
        keypoints_df = read_df(self.get_fp(KEYPOINTS_DIR), KEYPOINTS_SCHEMA)
        for func in funcs:
            keypoints_df = func(
                keypoints_df=keypoints_df,
                config=self.read_config(),
                metadata=self.read_metadata(),
            )
        # Write
        write_df(keypoints_df, self.get_fp(PREPROCESSED_DIR), KEYPOINTS_SCHEMA)

    @trace
    def extract_features(self, *, overwrite: bool) -> None:
        """Extracts features from the preprocessed dlc file."""
        # Overwrite check
        if not overwrite and self.get_fp(FEATURES_EXTRACTED_DIR).exists():
            log_file_exists(self.get_fp(FEATURES_EXTRACTED_DIR))
            return
        # Process
        keypoints_df = read_df(self.get_fp(KEYPOINTS_DIR), KEYPOINTS_SCHEMA)
        features_df = extract_features(
            keypoints_df=keypoints_df,
            config=self.read_config(),
            metadata=self.read_metadata(),
        )
        # Write
        features_df.write_parquet(self.get_fp(FEATURES_EXTRACTED_DIR))

    @trace
    def classify_behaviour(self, *, overwrite: bool) -> None:
        """Classify behaviours using trained models."""
        # Overwrite check
        if not overwrite and self.get_fp(BEHAVIOUR_PREDICTED_DIR).exists():
            log_file_exists(self.get_fp(BEHAVIOUR_PREDICTED_DIR))
            return
        # Process
        features_df = pl.read_parquet(self.get_fp(FEATURES_EXTRACTED_DIR))
        behaviour_df = classify_behaviour(
            features_df=features_df,
            config=self.read_config(),
            metadata=self.read_metadata(),
        )
        # Write
        write_df(
            behaviour_df,
            self.get_fp(BEHAVIOUR_PREDICTED_DIR),
            BEHAVIOUR_PREDICTED_SCHEMA,
        )

    @trace
    def export_behaviour(self, *, overwrite: bool) -> None:
        """Export predicted behaviours to scored behaviours."""
        # Overwrite check
        if not overwrite and self.get_fp(BEHAVIOUR_SCORED_DIR).exists():
            log_file_exists(self.get_fp(BEHAVIOUR_SCORED_DIR))
            return
        # Process
        # Read
        behaviour_predicted_df = read_df(
            self.get_fp(BEHAVIOUR_PREDICTED_DIR),
            BEHAVIOUR_PREDICTED_SCHEMA,
        )
        # Make list of bouts (behaviour and sub-behaviour)
        config = self.read_config()
        bouts_struct = [
            BoutStruct(
                behaviour=ProductionPointer.read_yaml(
                    model_config.clf_fp
                ).behaviour_name,
                sub_behaviour=model_config.user_defined,
            )
            for model_config in config.require_classify_behaviour()
        ]
        # Convert predicted to scored
        behaviour_scored_df = predicted_to_scored(behaviour_predicted_df, bouts_struct)
        # Write
        behaviour_scored_df.write_parquet(self.get_fp(BEHAVIOUR_SCORED_DIR))

    @trace
    def analyse(self, funcs: tuple[AnalyseFunc, ...]) -> None:
        """Analyse preprocessed keypoints data."""
        keypoints_df = read_df(self.get_fp(KEYPOINTS_DIR), KEYPOINTS_SCHEMA)
        vid_frame = _get_frame(self.get_fp(FORMATTED_VIDEO_DIR), self.read_metadata())
        config = self.read_config()
        metadata = self.read_metadata()
        for func in funcs:
            dst_dir = self.root_dir / ANALYSIS_DIR / func.__name__
            for result in func(keypoints_df, vid_frame, config, metadata):
                result.save(dst_dir)

    @trace
    def analyse_behaviour(self) -> None:
        """Analyse scored behaviours."""
        behaviour_df = read_df(
            self.get_fp(BEHAVIOUR_SCORED_DIR),
            BEHAVIOUR_SCORED_BASE,
        )
        dst_dir = self.root_dir / ANALYSIS_DIR / "analyse_behaviour"
        for result in analyse_behaviour(
            behaviour_df,
            self.read_config(),
            self.read_metadata(),
        ):
            result.save(dst_dir)

    @trace
    def combine_analysis(self) -> None:
        """Combine the experiment's analysis into a single df."""
        combine_analysis(
            name=self.name,
            analysis_combined_fp=self.get_fp(ANALYSIS_COMBINED_DIR),
            analysis_dir=self.root_dir / ANALYSIS_DIR,
        )
