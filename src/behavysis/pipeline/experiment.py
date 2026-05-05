"""Experiment class for processing a single experiment in the behavysis pipeline."""

import inspect
import logging
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np

from behavysis.constants import (
    ANALYSIS_DIR,
    FileExts,
    Folders,
)
from behavysis.models.process_result import ProcessResult, ProcessResultCollection
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

logger = logging.getLogger(__name__)


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
                f"  Tip: Check the experiment name matches your file names (without extension)."
            )
            raise ValueError(msg)

    def get_fp(self, folder: Folders | str) -> Path:
        """Returns the experiment's file path from the given folder."""
        if isinstance(folder, str):
            try:
                folder = Folders(folder)
            except ValueError as e:
                valid = "".join([f"\n    - {f.value}" for f in Folders])
                msg = (
                    f'Invalid folder: "{folder}"\n'
                    f"  Valid folders:{valid}"
                )
                raise ValueError(msg) from e
        file_ext: FileExts = getattr(FileExts, folder.name)
        return self.root_dir / folder.value / f"{self.name}.{file_ext.value}"

    def _run_funcs_with_filtered_kwargs(
        self, funcs: tuple[Callable, ...], **kwargs
    ) -> ProcessResultCollection:
        """Run functions with only the kwargs each function accepts.

        This allows the caller to pass all available kwargs without
        worrying about which function needs which parameters.
        """
        f_names_ls_msg = "".join([f"\n    - {f.__name__}" for f in funcs])
        logger.info("Processing experiment, %s, with:%s", self.name, f_names_ls_msg)
        results = ProcessResultCollection(experiment=self.name)
        for f in funcs:
            f_name = f.__name__
            sig = inspect.signature(f)
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            result = ProcessResult(process_name=f_name)
            try:
                f(**filtered_kwargs)
                result.mark_complete(success=True)
            except Exception as e:
                result.add_log(logging.ERROR, str(e))
                logger.debug(traceback.format_exc())
                result.mark_complete(success=False, error_message=str(e))
            results.results[f_name] = result
        results.mark_complete()
        logger.info(
            "Finished processing experiment, %s, with:%s", self.name, f_names_ls_msg
        )
        return results

    def update_configs(
        self, default_configs_fp: str, *, overwrite: str
    ) -> ProcessResultCollection:
        """Initialises the JSON config files with the given configurations."""
        return self._run_funcs_with_filtered_kwargs(
            (update_configs,),
            configs_fp=self.get_fp(Folders.CONFIGS),
            default_configs_fp=default_configs_fp,
            overwrite=overwrite,
        )

    def format_vid(self, *, overwrite: bool) -> ProcessResultCollection:
        """Formats the video with ffmpeg to fit the formatted configs."""
        return self._run_funcs_with_filtered_kwargs(
            (format_vid,),
            raw_vid_fp=self.get_fp(Folders.RAW_VID),
            formatted_vid_fp=self.get_fp(Folders.FORMATTED_VID),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    def run_dlc(
        self, gputouse: int | None, *, overwrite: bool
    ) -> ProcessResultCollection:
        """Run the DLC model on the formatted video."""
        return self._run_funcs_with_filtered_kwargs(
            (ma_dlc_run_single,),
            formatted_vid_fp=self.get_fp(Folders.FORMATTED_VID),
            keypoints_fp=self.get_fp(Folders.KEYPOINTS),
            configs_fp=self.get_fp(Folders.CONFIGS),
            gputouse=gputouse,
            overwrite=overwrite,
        )

    def calculate_parameters(
        self, funcs: tuple[Callable, ...]
    ) -> ProcessResultCollection:
        """Calculate parameters of the keypoints file."""
        return self._run_funcs_with_filtered_kwargs(
            funcs,
            keypoints_fp=self.get_fp(Folders.KEYPOINTS),
            configs_fp=self.get_fp(Folders.CONFIGS),
        )

    def preprocess(
        self, funcs: tuple[Callable, ...], *, overwrite: bool
    ) -> ProcessResultCollection:
        """Preprocessing pipeline for keypoints data."""
        results0 = self._run_funcs_with_filtered_kwargs(
            (df2df,),
            src_fp=self.get_fp(Folders.KEYPOINTS),
            dst_fp=self.get_fp(Folders.PREPROCESSED),
            overwrite=overwrite,
        )
        if not results0.results[df2df.__name__].success:
            return results0
        results1 = self._run_funcs_with_filtered_kwargs(
            funcs,
            src_fp=self.get_fp(Folders.PREPROCESSED),
            dst_fp=self.get_fp(Folders.PREPROCESSED),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=True,
        )
        return ProcessResultCollection(
            experiment=self.name, results={**results0.results, **results1.results}
        )

    def extract_features(self, *, overwrite: bool) -> ProcessResultCollection:
        """Extracts features from the preprocessed dlc file."""
        return self._run_funcs_with_filtered_kwargs(
            (extract_features,),
            keypoints_fp=self.get_fp(Folders.PREPROCESSED),
            features_fp=self.get_fp(Folders.FEATURES_EXTRACTED),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    def classify_behavs(self, *, overwrite: bool) -> ProcessResultCollection:
        """Classify behaviours using trained models."""
        return self._run_funcs_with_filtered_kwargs(
            (classify_behavs,),
            features_fp=self.get_fp(Folders.FEATURES_EXTRACTED),
            behavs_fp=self.get_fp(Folders.PREDICTED_BEHAVS),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    def export_behavs(self, *, overwrite: bool) -> ProcessResultCollection:
        """Export predicted behaviours to scored behaviours."""
        return self._run_funcs_with_filtered_kwargs(
            (predictedbehavs2scoredbehavs,),
            src_fp=self.get_fp(Folders.PREDICTED_BEHAVS),
            dst_fp=self.get_fp(Folders.SCORED_BEHAVS),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    def analyse(self, funcs: tuple[Callable, ...]) -> ProcessResultCollection:
        """Analyse preprocessed keypoints data."""
        return self._run_funcs_with_filtered_kwargs(
            funcs,
            keypoints_fp=self.get_fp(Folders.PREPROCESSED),
            formatted_vid_fp=self.get_fp(Folders.FORMATTED_VID),
            dst_dir=self.root_dir / ANALYSIS_DIR,
            configs_fp=self.get_fp(Folders.CONFIGS),
        )

    def analyse_behavs(self) -> ProcessResultCollection:
        """Analyse scored behaviours."""
        return self._run_funcs_with_filtered_kwargs(
            (analyse_behavs,),
            behavs_fp=self.get_fp(Folders.SCORED_BEHAVS),
            configs_fp=self.get_fp(Folders.CONFIGS),
            dst_dir=self.root_dir / ANALYSIS_DIR,
        )

    def combine_analysis(self) -> ProcessResultCollection:
        """Combine the experiment's analysis into a single df."""
        return self._run_funcs_with_filtered_kwargs(
            (combine_analysis,),
            analysis_combined_fp=self.get_fp(Folders.ANALYSIS_COMBINED),
            configs_fp=self.get_fp(Folders.CONFIGS),
            analysis_dir=self.root_dir / ANALYSIS_DIR,
            overwrite=True,
        )

    def evaluate_vid(self, *, overwrite: bool) -> ProcessResultCollection:
        """Generate annotated evaluation video."""
        return self._run_funcs_with_filtered_kwargs(
            (EvaluateVid.evaluate_vid,),
            formatted_vid_fp=self.get_fp(Folders.FORMATTED_VID),
            keypoints_fp=self.get_fp(Folders.PREPROCESSED),
            analysis_combined_fp=self.get_fp(Folders.ANALYSIS_COMBINED),
            eval_vid_fp=self.get_fp(Folders.EVALUATE_VID),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    def export2csv(
        self, src_dir: str, dst_dir: str | Path, *, overwrite: bool
    ) -> ProcessResultCollection:
        """Export dataframe to CSV."""
        return self._run_funcs_with_filtered_kwargs(
            (df2csv,),
            src_fp=self.get_fp(src_dir),
            dst_fp=Path(dst_dir) / f"{self.name}.csv",
            overwrite=overwrite,
        )
