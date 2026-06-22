"""Experiment class for processing a single experiment in the behavysis pipeline."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np

from behavysis.constants import (
    ANALYSIS_DIR,
    FileExts,
    Folders,
)
from behavysis.processes import (
    EvaluateVid,
    analyse_behavs,
    classify_behavs,
    combine_analysis,
    df2csv,
    df2df,
    extract_features,
    format_vid,
    ma_dlc_run_single,
    predictedbehavs2scoredbehavs,
    update_configs,
)
from behavysis.processes.calculate_params import CalculateParamsFunc
from behavysis.processes.preprocess import PreprocessFunc
from behavysis.utils.logger_utils import trace


class Experiment:
    """Behavysis Pipeline class for a single experiment."""

    name: str
    root_dir: Path

    def __init__(self, name: str, root_dir: str | Path) -> None:
        """Initialises the experiment with the given name and root directory."""
        root_dir = Path(root_dir)
        if not root_dir.is_dir():
            msg = (
                f'Project folder not found: "{root_dir}"\n'
                f"  Create a new project with: behavysis-make-project"
            )
            raise ValueError(msg)
        self.name = name
        self.root_dir = root_dir.resolve()
        file_exists_ls = [self.get_fp(f).is_file() for f in Folders]
        if not np.any(file_exists_ls):
            folders_ls_msg = "".join([f"\n    - {f.value}" for f in Folders])
            msg = (
                f'No files named "{name}" found in "{root_dir}".\n'
                f"  Expected files in one of these folders:{folders_ls_msg}\n"
                "  Tip: Check the experiment name matches your file names "
                "(without extension)."
            )
            raise ValueError(msg)

    def get_log_context(self) -> dict:
        """Log context for loguru context."""
        return {"experiment": str(self.root_dir)}

    def get_fp(self, folder: Folders | str) -> Path:
        """Returns the experiment's file path from the given folder."""
        if isinstance(folder, str):
            try:
                folder = Folders(folder)
            except ValueError as e:
                valid = "".join([f"\n    - {f.value}" for f in Folders])
                msg = f'Invalid folder: "{folder}"\n  Valid folders:{valid}'
                raise ValueError(msg) from e
        file_ext: FileExts = getattr(FileExts, folder.name)
        return self.root_dir / folder.value / f"{self.name}.{file_ext.value}"

    @trace
    def update_configs(
        self,
        default_configs_fp: str | Path,
        *,
        overwrite: Literal["user", "all"],
    ) -> None:
        """Initialises the JSON config files with the given configurations."""
        update_configs(
            configs_fp=self.get_fp(Folders.CONFIGS),
            default_configs_fp=Path(default_configs_fp),
            overwrite=overwrite,
        )

    @trace
    def format_vid(self, *, overwrite: bool) -> None:
        """Formats the video with ffmpeg to fit the formatted configs."""
        format_vid(
            raw_vid_fp=self.get_fp(Folders.RAW_VID),
            formatted_vid_fp=self.get_fp(Folders.FORMATTED_VID),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    @trace
    def run_dlc(self, gputouse: int | None, *, overwrite: bool) -> None:
        """Run the DLC model on the formatted video."""
        ma_dlc_run_single(
            formatted_vid_fp=self.get_fp(Folders.FORMATTED_VID),
            keypoints_fp=self.get_fp(Folders.KEYPOINTS),
            configs_fp=self.get_fp(Folders.CONFIGS),
            gputouse=gputouse,
            overwrite=overwrite,
        )

    @trace
    def calculate_parameters(self, funcs: tuple[CalculateParamsFunc, ...]) -> None:
        """Calculate parameters of the keypoints file."""
        for func in funcs:
            func(
                keypoints_fp=self.get_fp(Folders.KEYPOINTS),
                configs_fp=self.get_fp(Folders.CONFIGS),
            )

    @trace
    def preprocess(self, funcs: tuple[PreprocessFunc, ...], *, overwrite: bool) -> None:
        """Preprocessing pipeline for keypoints data."""
        df2df(
            src_fp=self.get_fp(Folders.KEYPOINTS),
            dst_fp=self.get_fp(Folders.PREPROCESSED),
            overwrite=overwrite,
        )
        for func in funcs:
            func(
                src_fp=self.get_fp(Folders.PREPROCESSED),
                dst_fp=self.get_fp(Folders.PREPROCESSED),
                configs_fp=self.get_fp(Folders.CONFIGS),
                overwrite=True,
            )

    @trace
    def extract_features(self, *, overwrite: bool) -> None:
        """Extracts features from the preprocessed dlc file."""
        extract_features(
            keypoints_fp=self.get_fp(Folders.PREPROCESSED),
            features_fp=self.get_fp(Folders.FEATURES_EXTRACTED),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    @trace
    def classify_behavs(self, *, overwrite: bool) -> None:
        """Classify behaviours using trained models."""
        classify_behavs(
            features_fp=self.get_fp(Folders.FEATURES_EXTRACTED),
            behavs_fp=self.get_fp(Folders.PREDICTED_BEHAVS),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    @trace
    def export_behavs(self, *, overwrite: bool) -> None:
        """Export predicted behaviours to scored behaviours."""
        predictedbehavs2scoredbehavs(
            src_fp=self.get_fp(Folders.PREDICTED_BEHAVS),
            dst_fp=self.get_fp(Folders.SCORED_BEHAVS),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    @trace
    def analyse(self, funcs: tuple[Callable, ...]) -> None:
        """Analyse preprocessed keypoints data."""
        for func in funcs:
            func(
                keypoints_fp=self.get_fp(Folders.PREPROCESSED),
                formatted_vid_fp=self.get_fp(Folders.FORMATTED_VID),
                dst_dir=self.root_dir / ANALYSIS_DIR,
                configs_fp=self.get_fp(Folders.CONFIGS),
            )

    @trace
    def analyse_behavs(self) -> None:
        """Analyse scored behaviours."""
        analyse_behavs(
            behavs_fp=self.get_fp(Folders.SCORED_BEHAVS),
            configs_fp=self.get_fp(Folders.CONFIGS),
            dst_dir=self.root_dir / ANALYSIS_DIR,
        )

    @trace
    def combine_analysis(self) -> None:
        """Combine the experiment's analysis into a single df."""
        combine_analysis(
            analysis_combined_fp=self.get_fp(Folders.ANALYSIS_COMBINED),
            configs_fp=self.get_fp(Folders.CONFIGS),
            analysis_dir=self.root_dir / ANALYSIS_DIR,
            overwrite=True,
        )

    @trace
    def evaluate_vid(self, *, overwrite: bool) -> None:
        """Generate annotated evaluation video."""
        EvaluateVid.evaluate_vid(
            formatted_vid_fp=self.get_fp(Folders.FORMATTED_VID),
            keypoints_fp=self.get_fp(Folders.PREPROCESSED),
            analysis_combined_fp=self.get_fp(Folders.ANALYSIS_COMBINED),
            eval_vid_fp=self.get_fp(Folders.EVALUATE_VID),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    @trace
    def export2csv(self, src_dir: str, dst_dir: str | Path, *, overwrite: bool) -> None:
        """Export dataframe to CSV."""
        df2csv(
            src_fp=self.get_fp(src_dir),
            dst_fp=Path(dst_dir) / f"{self.name}.csv",
            overwrite=overwrite,
        )
