"""Experiment class for processing a single experiment in the behavysis pipeline."""

import shutil
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from behavysis.behaviour_classifier import ClassifierPaths
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
    ExtractFeaturesFunc,
    PreprocessFunc,
    classify_behaviour,
    combine_analysis,
    dlc_run_ma,
    format_video,
    get_video_metadata,
)
from behavysis.models import BoutStruct, ExperimentConfig, ExperimentMetadata
from behavysis.schemas import (
    BEHAVIOUR_PREDICTED_SCHEMA,
    KEYPOINTS_SCHEMA,
    read_df,
    write_df,
)
from behavysis.transforms import predicted_to_scored
from behavysis.utils import has_output_files, missing_input_files, trace


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

    def get_features_fp(self, feature_set: str) -> Path:
        """Returns the experiment's features file path for a named feature set."""
        return (
            self.root_dir
            / FEATURES_EXTRACTED_DIR
            / feature_set
            / f"{self.name}.{STAGES[FEATURES_EXTRACTED_DIR]}"
        )

    def read_config(self) -> ExperimentConfig:
        """Returns the experiment's config."""
        return ExperimentConfig.read_yaml(self.get_fp(CONFIG_DIR))

    def read_metadata(self) -> ExperimentMetadata:
        """Returns the experiment's metadata."""
        if not self.get_fp(METADATA_DIR).exists():
            return ExperimentMetadata()
        return ExperimentMetadata.read_yaml(self.get_fp(METADATA_DIR))

    def write_metadata(self, metadata: ExperimentMetadata) -> None:
        """Save the experiment's metadata to disk."""
        self.get_fp(METADATA_DIR).parent.mkdir(parents=True, exist_ok=True)
        metadata.write_yaml(self.get_fp(METADATA_DIR))

    @trace
    def update_config(self, default_config_fp: Path, *, overwrite: bool) -> None:
        """Copy the default configs to this project."""
        if not overwrite and has_output_files(self.get_fp(CONFIG_DIR)):
            return
        ExperimentConfig.read_yaml(default_config_fp)
        self.get_fp(CONFIG_DIR).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(default_config_fp, self.get_fp(CONFIG_DIR))

    @trace
    def format_video(self, *, overwrite: bool) -> None:
        """Formats the video with ffmpeg to fit the formatted config."""
        if not overwrite and has_output_files(self.get_fp(FORMATTED_VIDEO_DIR)):
            return
        if missing_input_files(self.get_fp(RAW_VIDEO_DIR)):
            return
        format_video(
            raw_vid_fp=self.get_fp(RAW_VIDEO_DIR),
            formatted_vid_fp=self.get_fp(FORMATTED_VIDEO_DIR),
            config=self.read_config(),
        )
        # Update metadata
        self.get_video_metadata()

    @trace
    def get_video_metadata(self) -> None:
        """Get vid metadata and save."""
        metadata = self.read_metadata()
        metadata.raw_video = get_video_metadata(self.get_fp(RAW_VIDEO_DIR))
        metadata.formatted_video = get_video_metadata(self.get_fp(FORMATTED_VIDEO_DIR))
        self.write_metadata(metadata)

    @trace
    def run_dlc(self, gputouse: int | None, *, overwrite: bool) -> None:
        """Run the DLC model on the formatted video."""
        if not overwrite and has_output_files(self.get_fp(KEYPOINTS_DIR)):
            return
        if missing_input_files(self.get_fp(FORMATTED_VIDEO_DIR)):
            return
        dlc_run_ma(
            vid_fp=self.get_fp(FORMATTED_VIDEO_DIR),
            keypoints_fp=self.get_fp(KEYPOINTS_DIR),
            config=self.read_config(),
            gputouse=gputouse,
        )

    @trace
    def calculate_parameters(self, funcs: tuple[CalculateParametersFunc, ...]) -> None:
        """Calculate parameters of the keypoints file."""
        if missing_input_files(self.get_fp(KEYPOINTS_DIR)):
            return
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
        if not overwrite and has_output_files(self.get_fp(PREPROCESSED_DIR)):
            return
        if missing_input_files(self.get_fp(KEYPOINTS_DIR)):
            return
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
    def extract_features(
        self, funcs: tuple[ExtractFeaturesFunc, ...], *, overwrite: bool
    ) -> None:
        """Extract features for each configured feature set."""
        if missing_input_files(self.get_fp(PREPROCESSED_DIR)):
            return
        keypoints_df = read_df(self.get_fp(PREPROCESSED_DIR), KEYPOINTS_SCHEMA)
        config = self.read_config()
        metadata = self.read_metadata()
        for func in funcs:
            out_fp = self.get_features_fp(func.__name__)
            if not overwrite and has_output_files(out_fp):
                continue
            features_df = func(
                keypoints_df=keypoints_df,
                config=config,
                metadata=metadata,
            )
            out_fp.parent.mkdir(parents=True, exist_ok=True)
            write_df(features_df, out_fp)

    @trace
    def classify_behaviour(self, *, overwrite: bool) -> None:
        """Classify behaviours using trained models."""
        if not overwrite and has_output_files(self.get_fp(BEHAVIOUR_PREDICTED_DIR)):
            return
        behaviour_df_ls = []
        for model_config in self.read_config().require_classify_behaviour():
            contract_fp = model_config.contract_fp
            clf = ClassifierPaths(contract_fp)
            if missing_input_files(self.get_features_fp(clf.contract().feature_set)):
                continue
            features_df = read_df(self.get_features_fp(clf.contract().feature_set))
            behaviour_df_ls.append(classify_behaviour(contract_fp, features_df))
        write_df(
            pl.concat(behaviour_df_ls)
            if behaviour_df_ls
            else pl.DataFrame(schema=BEHAVIOUR_PREDICTED_SCHEMA),
            self.get_fp(BEHAVIOUR_PREDICTED_DIR),
            BEHAVIOUR_PREDICTED_SCHEMA,
        )

    @trace
    def export_behaviour(self, *, overwrite: bool) -> None:
        """Export predicted behaviours to scored behaviours."""
        if not overwrite and has_output_files(self.get_fp(BEHAVIOUR_SCORED_DIR)):
            return
        if missing_input_files(self.get_fp(BEHAVIOUR_PREDICTED_DIR)):
            return
        behaviour_predicted_df = read_df(
            self.get_fp(BEHAVIOUR_PREDICTED_DIR), BEHAVIOUR_PREDICTED_SCHEMA
        )
        config = self.read_config()
        bouts_struct = [
            BoutStruct(
                behaviour=ClassifierPaths(model_config.contract_fp)
                .contract()
                .behaviour_name,
                sub_behaviour=model_config.sub_behaviour,
            )
            for model_config in config.require_classify_behaviour()
        ]
        behaviour_scored_df = predicted_to_scored(behaviour_predicted_df, bouts_struct)
        self.get_fp(BEHAVIOUR_SCORED_DIR).parent.mkdir(parents=True, exist_ok=True)
        behaviour_scored_df.write_parquet(self.get_fp(BEHAVIOUR_SCORED_DIR))

    @trace
    def analyse(self, funcs: tuple[AnalyseFunc, ...]) -> None:
        """Analyse preprocessed keypoints and/or scored behaviours."""
        config = self.read_config()
        metadata = self.read_metadata()
        kwargs: dict[str, object] = {}
        if self.get_fp(PREPROCESSED_DIR).is_file():
            kwargs["keypoints_df"] = read_df(
                self.get_fp(PREPROCESSED_DIR), KEYPOINTS_SCHEMA
            )
        if self.get_fp(FORMATTED_VIDEO_DIR).is_file():
            kwargs["vid_frame"] = _get_frame(self.get_fp(FORMATTED_VIDEO_DIR), metadata)
        if self.get_fp(BEHAVIOUR_SCORED_DIR).is_file():
            kwargs["behaviour_df"] = read_df(self.get_fp(BEHAVIOUR_SCORED_DIR))
        for func in funcs:
            dst_dir = self.root_dir / ANALYSIS_DIR / func.__name__
            for result in func(config, metadata, **kwargs):
                result.save(dst_dir)

    @trace
    def combine_analysis(self) -> None:
        """Combine the experiment's analysis into a single df."""
        combine_analysis(
            name=self.name,
            analysis_combined_fp=self.get_fp(ANALYSIS_COMBINED_DIR),
            analysis_dir=self.root_dir / ANALYSIS_DIR,
        )
