"""Experiment class for processing a single experiment in the behavysis pipeline."""

import logging
import os
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np

from behavysis.constants import (
    ANALYSIS_DIR,
    FileExts,
    Folders,
)
from behavysis.models.experiment_configs import AutoConfigs, ExperimentConfigs
from behavysis.processes.analyse_behavs import AnalyseBehavs
from behavysis.processes.classify_behavs import ClassifyBehavs
from behavysis.processes.combine_analysis import CombineAnalysis
from behavysis.processes.evaluate_vid import EvaluateVid
from behavysis.processes.export import Export
from behavysis.processes.extract_features import ExtractFeatures
from behavysis.processes.format_vid import FormatVid
from behavysis.processes.run_dlc import RunDLC
from behavysis.processes.update_configs import UpdateConfigs
from behavysis.utils.diagnostics_utils import ProcessResult, ProcessResultCollection

logger = logging.getLogger(__name__)


class Experiment:
    """Behavysis Pipeline class for a single experiment.

    Encompasses the entire process including:
    - Raw mp4 file import.
    - mp4 file formatting (px and fps).
    - DLC keypoints inference.
    - Feature wrangling (start time detection, more features like average body position).
    - Interpretable behaviour results.
    - Other quantitative analysis.
    """

    def __init__(self, name: str, root_dir: str) -> None:
        """Make a Experiment instance."""
        if not os.path.isdir(root_dir):
            raise ValueError(
                f'Cannot find the project folder named "{root_dir}".\n'
                "Please specify a folder that exists."
            )
        self.name = name
        self.root_dir = os.path.abspath(root_dir)
        file_exists_ls = [os.path.isfile(self.get_fp(f)) for f in Folders]
        if not np.any(file_exists_ls):
            folders_ls_msg = "".join([f"\n    - {f.value}" for f in Folders])
            raise ValueError(
                f'No files named "{name}" exist in "{root_dir}".\n'
                f"Please specify a file in one of these folders:{folders_ls_msg}"
            )

    def get_fp(self, folder: Folders | str) -> str:
        """Returns the experiment's file path from the given folder."""
        if isinstance(folder, str):
            try:
                folder = Folders(folder)
            except ValueError:
                valid = "".join([f"\n    - {f.value}" for f in Folders])
                raise ValueError(
                    f"{folder} is not a valid folder. Valid folders:{valid}"
                )
        file_ext: FileExts = getattr(FileExts, folder.name)
        return os.path.join(
            self.root_dir, folder.value, f"{self.name}.{file_ext.value}"
        )

    def _analysis_dir(self) -> str:
        """Returns the analysis directory path for this experiment."""
        return os.path.join(self.root_dir, ANALYSIS_DIR)

    def _proc_scaff(
        self, funcs: tuple[Callable, ...], *args: Any, **kwargs: Any
    ) -> ProcessResultCollection:
        """All processing runs through here."""
        f_names_ls_msg = "".join([f"\n    - {f.__name__}" for f in funcs])
        logger.info(f"Processing experiment, {self.name}, with:{f_names_ls_msg}")
        results = ProcessResultCollection(experiment=self.name)
        for f in funcs:
            f_name = f.__name__
            result = ProcessResult(process_name=f_name)
            try:
                f(*args, **kwargs)
                result.mark_complete(success=True)
            except Exception as e:
                result.add_log(logging.ERROR, str(e))
                logger.debug(traceback.format_exc())
                result.mark_complete(success=False, error_message=str(e))
            results.results[f_name] = result
        logger.info(
            f"Finished processing experiment, {self.name}, with:{f_names_ls_msg}"
        )
        return results

    def update_configs(
        self, default_configs_fp: str, overwrite: str
    ) -> ProcessResultCollection:
        """Initialises the JSON config files with the given configurations."""
        return self._proc_scaff(
            (UpdateConfigs.update_configs,),
            configs_fp=self.get_fp(Folders.CONFIGS),
            default_configs_fp=default_configs_fp,
            overwrite=overwrite,
        )

    def format_vid(self, *, overwrite: bool) -> ProcessResultCollection:
        """Formats the video with ffmpeg to fit the formatted configs."""
        return self._proc_scaff(
            (FormatVid.format_vid,),
            raw_vid_fp=self.get_fp(Folders.RAW_VID),
            formatted_vid_fp=self.get_fp(Folders.FORMATTED_VID),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    def get_vid_metadata(self) -> ProcessResultCollection:
        """Gets the video metadata for the raw and formatted video files."""
        return self._proc_scaff(
            (FormatVid.get_vids_metadata,),
            raw_vid_fp=self.get_fp(Folders.RAW_VID),
            formatted_vid_fp=self.get_fp(Folders.FORMATTED_VID),
            configs_fp=self.get_fp(Folders.CONFIGS),
        )

    def run_dlc(
        self, *, gputouse: int | None, overwrite: bool
    ) -> ProcessResultCollection:
        """Run the DLC model on the formatted video."""
        return self._proc_scaff(
            (RunDLC.ma_dlc_run_single,),
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
        return self._proc_scaff(
            funcs,
            keypoints_fp=self.get_fp(Folders.KEYPOINTS),
            configs_fp=self.get_fp(Folders.CONFIGS),
        )

    def collate_auto_configs(self) -> ProcessResultCollection:
        """Collates the auto-configs of the experiment into the main configs file."""
        result = ProcessResult(process_name="reading_configs")
        try:
            configs = ExperimentConfigs.read_json(self.get_fp(Folders.CONFIGS))
            result.add_log(logging.DEBUG, "Reading configs file.")
            result.mark_complete(success=True)
        except FileNotFoundError:
            result.add_log(logging.ERROR, "no configs file found.")
            result.mark_complete(success=False, error_message="no configs file found.")
            return ProcessResultCollection(
                experiment=self.name, results={"reading_configs": result}
            )
        data = {}
        for field_key_ls in AutoConfigs.get_field_names():
            value = configs.auto
            for key in field_key_ls:
                value = getattr(value, key)
            data["_".join(field_key_ls)] = value
        return ProcessResultCollection(
            experiment=self.name, results={"reading_configs": result, "data": data}
        )

    def preprocess(
        self, funcs: tuple[Callable, ...], *, overwrite: bool
    ) -> ProcessResultCollection:
        """Preprocessing pipeline for keypoints data."""
        results0 = self._proc_scaff(
            (Export.df2df,),
            src_fp=self.get_fp(Folders.KEYPOINTS),
            dst_fp=self.get_fp(Folders.PREPROCESSED),
            overwrite=overwrite,
        )
        if not results0.results[Export.df2df.__name__].success:
            return results0
        results1 = self._proc_scaff(
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
        return self._proc_scaff(
            (ExtractFeatures.extract_features,),
            keypoints_fp=self.get_fp(Folders.PREPROCESSED),
            features_fp=self.get_fp(Folders.FEATURES_EXTRACTED),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    def classify_behavs(self, *, overwrite: bool) -> ProcessResultCollection:
        """Classify behaviours using trained models."""
        return self._proc_scaff(
            (ClassifyBehavs.classify_behavs,),
            features_fp=self.get_fp(Folders.FEATURES_EXTRACTED),
            behavs_fp=self.get_fp(Folders.PREDICTED_BEHAVS),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    def export_behavs(self, *, overwrite: bool) -> ProcessResultCollection:
        """Export predicted behaviours to scored behaviours."""
        return self._proc_scaff(
            (Export.predictedbehavs2scoredbehavs,),
            src_fp=self.get_fp(Folders.PREDICTED_BEHAVS),
            dst_fp=self.get_fp(Folders.SCORED_BEHAVS),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    def analyse(self, funcs: tuple[Callable, ...]) -> ProcessResultCollection:
        """Analyse preprocessed DLC data."""
        return self._proc_scaff(
            funcs,
            keypoints_fp=self.get_fp(Folders.PREPROCESSED),
            formatted_vid_fp=self.get_fp(Folders.FORMATTED_VID),
            dst_dir=self._analysis_dir(),
            configs_fp=self.get_fp(Folders.CONFIGS),
        )

    def analyse_behavs(self) -> ProcessResultCollection:
        """Analyse scored behaviours."""
        return self._proc_scaff(
            (AnalyseBehavs.analyse_behavs,),
            behavs_fp=self.get_fp(Folders.SCORED_BEHAVS),
            configs_fp=self.get_fp(Folders.CONFIGS),
            dst_dir=self._analysis_dir(),
        )

    def combine_analysis(self) -> ProcessResultCollection:
        """Combine the experiment's analysis into a single df."""
        return self._proc_scaff(
            (CombineAnalysis.combine_analysis,),
            analysis_combined_fp=self.get_fp(Folders.ANALYSIS_COMBINED),
            configs_fp=self.get_fp(Folders.CONFIGS),
            analysis_dir=self._analysis_dir(),
            overwrite=True,
        )

    def evaluate_vid(self, *, overwrite: bool) -> ProcessResultCollection:
        """Generate annotated evaluation video."""
        return self._proc_scaff(
            (EvaluateVid.evaluate_vid,),
            formatted_vid_fp=self.get_fp(Folders.FORMATTED_VID),
            keypoints_fp=self.get_fp(Folders.PREPROCESSED),
            analysis_combined_fp=self.get_fp(Folders.ANALYSIS_COMBINED),
            eval_vid_fp=self.get_fp(Folders.EVALUATE_VID),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    def export2csv(
        self, src_dir: str, dst_dir: str, *, overwrite: bool
    ) -> ProcessResultCollection:
        """Export dataframe to CSV."""
        return self._proc_scaff(
            (Export.df2csv,),
            src_fp=self.get_fp(src_dir),
            dst_fp=os.path.join(dst_dir, f"{self.name}.csv"),
            overwrite=overwrite,
        )
