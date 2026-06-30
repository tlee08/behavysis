"""Experiment class for processing a single experiment in the behavysis pipeline."""

from pathlib import Path
from typing import Literal

import numpy as np

from behavysis.constants import (
    ANALYSIS_COMBINED_DIR,
    ANALYSIS_DIR,
    CONFIG_DIR,
    FORMATTED_VIDEO_DIR,
    KEYPOINTS_DIR,
    RAW_VIDEO_DIR,
    STAGES,
)
from behavysis.constants.pipeline import (
    FEATURES_EXTRACTED_DIR,
    PREDICTED_BEHAVIOUR_DIR,
    PREPROCESSED_DIR,
    SCORED_BEHAVIOUR_DIR,
)
from behavysis.funcs import (
    AnalyseFunc,
    CalculateParamsFunc,
    PreprocessFunc,
    analyse_behaviour,
    classify_behaviour,
    combine_analysis,
    df2csv,
    df2df,
    extract_features,
    format_video,
    ma_dlc_run_single,
    predictedbehaviour2scoredbehaviour,
    update_config,
)
from behavysis.utils.logger_utils import trace


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

    def get_log_context(self) -> dict:
        """Log context for loguru context."""
        return {"experiment": str(self.name)}

    def get_fp(self, folder: str) -> Path:
        """Returns the experiment's file path from the given folder."""
        return self.root_dir / folder / f"{self.name}.{STAGES[folder]}"

    @trace
    def update_config(
        self,
        default_config_fp: str | Path,
        *,
        overwrite: Literal["user", "all"],
    ) -> None:
        """Initialises the JSON config files with the given configurations."""
        update_config(
            config_fp=self.get_fp(CONFIG_DIR),
            default_config_fp=Path(default_config_fp),
            overwrite=overwrite,
        )

    @trace
    def format_video(self, *, overwrite: bool) -> None:
        """Formats the video with ffmpeg to fit the formatted config."""
        format_video(
            raw_vid_fp=self.get_fp(RAW_VIDEO_DIR),
            formatted_vid_fp=self.get_fp(FORMATTED_VIDEO_DIR),
            config_fp=self.get_fp(CONFIG_DIR),
            overwrite=overwrite,
        )

    @trace
    def run_dlc(self, gputouse: int | None, *, overwrite: bool) -> None:
        """Run the DLC model on the formatted video."""
        ma_dlc_run_single(
            vid_fp=self.get_fp(FORMATTED_VIDEO_DIR),
            keypoints_fp=self.get_fp(KEYPOINTS_DIR),
            config_fp=self.get_fp(CONFIG_DIR),
            gputouse=gputouse,
            overwrite=overwrite,
        )

    @trace
    def calculate_parameters(self, funcs: tuple[CalculateParamsFunc, ...]) -> None:
        """Calculate parameters of the keypoints file."""
        for func in funcs:
            func(
                keypoints_fp=self.get_fp(KEYPOINTS_DIR),
                config_fp=self.get_fp(CONFIG_DIR),
            )

    @trace
    def preprocess(self, funcs: tuple[PreprocessFunc, ...], *, overwrite: bool) -> None:
        """Preprocessing pipeline for keypoints data."""
        df2df(
            src_fp=self.get_fp(KEYPOINTS_DIR),
            dst_fp=self.get_fp(PREPROCESSED_DIR),
            overwrite=overwrite,
        )
        for func in funcs:
            func(
                src_fp=self.get_fp(PREPROCESSED_DIR),
                dst_fp=self.get_fp(PREPROCESSED_DIR),
                config_fp=self.get_fp(CONFIG_DIR),
                overwrite=True,
            )

    @trace
    def extract_features(self, *, overwrite: bool) -> None:
        """Extracts features from the preprocessed dlc file."""
        extract_features(
            keypoints_fp=self.get_fp(PREPROCESSED_DIR),
            features_fp=self.get_fp(FEATURES_EXTRACTED_DIR),
            config_fp=self.get_fp(CONFIG_DIR),
            overwrite=overwrite,
        )

    @trace
    def classify_behaviour(self, *, overwrite: bool) -> None:
        """Classify behaviours using trained models."""
        classify_behaviour(
            features_fp=self.get_fp(FEATURES_EXTRACTED_DIR),
            behaviour_fp=self.get_fp(PREDICTED_BEHAVIOUR_DIR),
            config_fp=self.get_fp(CONFIG_DIR),
            overwrite=overwrite,
        )

    @trace
    def export_behaviour(self, *, overwrite: bool) -> None:
        """Export predicted behaviours to scored behaviours."""
        predictedbehaviour2scoredbehaviour(
            src_fp=self.get_fp(PREDICTED_BEHAVIOUR_DIR),
            dst_fp=self.get_fp(SCORED_BEHAVIOUR_DIR),
            config_fp=self.get_fp(CONFIG_DIR),
            overwrite=overwrite,
        )

    @trace
    def analyse(self, funcs: tuple[AnalyseFunc, ...]) -> None:
        """Analyse preprocessed keypoints data."""
        for func in funcs:
            func(
                keypoints_fp=self.get_fp(PREPROCESSED_DIR),
                formatted_vid_fp=self.get_fp(FORMATTED_VIDEO_DIR),
                dst_dir=self.root_dir / ANALYSIS_DIR,
                config_fp=self.get_fp(CONFIG_DIR),
            )

    @trace
    def analyse_behaviour(self) -> None:
        """Analyse scored behaviours."""
        analyse_behaviour(
            behaviour_fp=self.get_fp(SCORED_BEHAVIOUR_DIR),
            config_fp=self.get_fp(CONFIG_DIR),
            dst_dir=self.root_dir / ANALYSIS_DIR,
        )

    @trace
    def combine_analysis(self) -> None:
        """Combine the experiment's analysis into a single df."""
        combine_analysis(
            analysis_combined_fp=self.get_fp(ANALYSIS_COMBINED_DIR),
            config_fp=self.get_fp(CONFIG_DIR),
            analysis_dir=self.root_dir / ANALYSIS_DIR,
            overwrite=True,
        )

    @trace
    def export2csv(self, src_dir: str, dst_dir: str | Path, *, overwrite: bool) -> None:
        """Export dataframe to CSV."""
        df2csv(
            src_fp=self.get_fp(src_dir),
            dst_fp=Path(dst_dir) / f"{self.name}.csv",
            overwrite=overwrite,
        )
