"""
_summary_
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Callable

import numpy as np

from behavysis_pipeline.constants import (
    ANALYSIS_DIR,
    STR_DIV,
    FileExts,
    Folders,
)
from behavysis_pipeline.processes.analyse_behaviours import AnalyseBehaviours
from behavysis_pipeline.processes.analyse_combine import AnalyseCombine
from behavysis_pipeline.processes.classify_behavs import ClassifyBehavs
from behavysis_pipeline.processes.evaluate_vid import EvaluateVid
from behavysis_pipeline.processes.export import Export
from behavysis_pipeline.processes.extract_features import ExtractFeatures
from behavysis_pipeline.processes.run_dlc import RunDLC
from behavysis_pipeline.processes.update_configs import UpdateConfigs
from behavysis_pipeline.utils.diagnostics_utils import success_msg
from behavysis_pipeline.utils.logging_utils import init_logger
from behavysis_pipeline.utils.misc_utils import enum2tuple


class Experiment:
    """
    Behavysis Pipeline class for a single experiment.

    Encompasses the entire process including:
    - Raw mp4 file import.
    - mp4 file formatting (px and fps).
    - DLC keypoints inference.
    - Feature wrangling (start time detection, more features like average body position).
    - Interpretable behaviour results.
    - Other quantitative analysis.

    Parameters
    ----------
    name : str
        _description_
    root_dir : str
        _description_

    Raises
    ------
    ValueError
        ValueError: `root_dir` does not exist or `name` does not exist in the `root_dir` folder.
    """

    logger = init_logger(__name__)

    def __init__(self, name: str, root_dir: str) -> None:
        """
        Make a Experiment instance.
        """
        # Assertion: root_dir mus† exist
        if not os.path.isdir(root_dir):
            raise ValueError(
                f'Cannot find the project folder named "{root_dir}".\n' "Please specify a folder that exists."
            )
        # Setting up instance variables
        self.name = name
        self.root_dir = os.path.abspath(root_dir)
        # Assertion: name must correspond to at least one file in root_dir
        file_exists_ls = [os.path.isfile(self.get_fp(f)) for f in Folders]
        if not np.any(file_exists_ls):
            raise ValueError(
                f'No files named "{name}" exist in "{root_dir}".\n'
                f'Please specify a file that exists in "{root_dir}", in one of the'
                " following folder WITH the correct file extension name:\n"
                "    - "
                "\n    - ".join(enum2tuple(Folders))
            )

    #####################################################################
    #               GET/CHECK FILEPATH METHODS
    #####################################################################

    def get_fp(self, _folder: Enum | str) -> str:
        """
        Returns the experiment's file path from the given folder.

        Parameters
        ----------
        folder_str : str
            The folder to return the experiment document's filepath for.

        Returns
        -------
        str
            The experiment document's filepath.

        Raises
        ------
        ValueError
            ValueError: Folder name is not valid. Refer to FOLDERS constant for valid folder names.
        """
        # Getting folder as Enum
        # If folder is str, converting to Enum
        if isinstance(_folder, str):
            try:
                folder = Folders(_folder)
            except ValueError:
                # if folder_str not in Folders enum
                raise ValueError(
                    f'"{_folder}" is not a valid experiment folder name.\n'
                    "Please only specify one of the following folders:\n"
                    "    - "
                    "\n    - ".join([f.value for f in Folders])
                )
        else:
            # Otherwise, using given Enum
            folder = _folder
        # Getting file extension from enum
        file_ext: FileExts = getattr(FileExts, folder.name)
        # Getting experiment filepath for given folder
        fp = os.path.join(self.root_dir, folder.value, f"{self.name}{file_ext.value}")
        # Returning filepath
        return fp

    #####################################################################
    #               EXPERIMENT PROCESSING SCAFFOLD METHODS
    #####################################################################

    def _process_scaffold(
        self,
        funcs: tuple[Callable, ...],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, str]:
        """
        All processing runs through here.
        This method ensures that the stdout and diagnostics dict are correctly generated.

        Parameters
        ----------
        funcs : tuple[Callable, ...]
            List of functions.

        Returns
        -------
        dict[str, str]
            Diagnostics dictionary, with description of each function's outcome.

        Notes
        -----
        Each func in `funcs` is called in the form:
        ```
        func(*args, **kwargs)
        ```
        """
        self.logger.info(f"Processing experiment: {self.name}")
        # Setting up diagnostics dict
        dd = {"experiment": self.name}
        # Running functions and saving outcome to diagnostics dict
        for f in funcs:
            # Running each func and saving outcome
            try:
                dd[f.__name__] = f(*args, **kwargs)
                dd[f.__name__] += f"SUCCESS: {success_msg()}\n"
            except Exception as e:
                dd[f.__name__] = f"ERROR: {e}"
            self.logger.info(f"{f.__name__}: {dd[f.__name__]}")
        self.logger.info(STR_DIV)
        return dd

    #####################################################################
    #                        CONFIG FILE METHODS
    #####################################################################

    def update_configs(self, default_configs_fp: str, overwrite: str) -> dict:
        """
        Initialises the JSON config files with the given configurations in `configs`.
        It can be specified whether or not to overwrite existing configuration values.

        Parameters
        ----------
        default_configs_fp : str
            The JSON configs filepath to add/overwrite to the experiment's current configs file.
        overwrite : {"set", "reset"}
            Specifies how to overwrite existing configurations.
            If `add`, only parameters in `configs` not already in the config files are added.
            If `set`, all parameters in `configs` are set in the config files (overwriting).
            If `reset`, the config files are completely replaced by `configs`.

        Returns
        -------
        dict
            Diagnostics dictionary, with description of each function's outcome.
        """
        return self._process_scaffold(
            (UpdateConfigs.update_configs,),
            configs_fp=self.get_fp(Folders.CONFIGS),
            default_configs_fp=default_configs_fp,
            overwrite=overwrite,
        )

    #####################################################################
    #                    FORMATTING VIDEO METHODS
    #####################################################################

    def format_vid(self, funcs: tuple[Callable, ...], overwrite: bool) -> dict:
        """
        Formats the video with ffmpeg to fit the formatted configs (e.g. fps and resolution_px).
        Once the formatted video is produced, the configs dict and *configs.json file are
        updated with the formatted video's metadata.

        Parameters
        ----------
        funcs : tuple[Callable, ...]
            _description_
        overwrite : bool
            Whether to overwrite the output file (if it exists).

        Returns
        -------
        dict
            Diagnostics dictionary, with description of each function's outcome.

        Notes
        -----
        Can call any methods from `FormatVid`.
        """
        return self._process_scaffold(
            funcs,
            in_fp=self.get_fp(Folders.RAW_VID),
            out_fp=self.get_fp(Folders.FORMATTED_VID),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    #####################################################################
    #                        DLC METHODS
    #####################################################################

    def run_dlc(self, gputouse: int | None, overwrite: bool) -> dict:
        """
        Run the DLC model on the formatted video to generate a DLC annotated video
        and DLC h5 file for all experiments.

        Parameters
        ----------
        gputouse : int
            _description_
        overwrite : bool
            Whether to overwrite the output file (if it exists).

        Returns
        -------
        dict
            Diagnostics dictionary, with description of each function's outcome.

        Notes
        -----
        Can call any methods from `RunDLC`.
        """
        return self._process_scaffold(
            (RunDLC.ma_dlc_analyse_single,),
            vid_fp=self.get_fp(Folders.FORMATTED_VID),
            out_fp=self.get_fp(Folders.DLC),
            configs_fp=self.get_fp(Folders.CONFIGS),
            gputouse=gputouse,
            overwrite=overwrite,
        )

    def calculate_parameters(self, funcs: tuple[Callable, ...]) -> dict:
        """
        A pipeline to calculate the parameters of the raw DLC file, which will
        assist in preprocessing the DLC data.

        Parameters
        ----------
        funcs : Tuple[Callable, ...]
            _description_

        Returns
        -------
        Dict
            Diagnostics dictionary, with description of each function's outcome.

        Notes
        -----
        Can call any methods from `CalculateParams`.
        """
        return self._process_scaffold(
            funcs,
            dlc_fp=self.get_fp(Folders.DLC),
            configs_fp=self.get_fp(Folders.CONFIGS),
        )

    def preprocess(self, funcs: tuple[Callable, ...], overwrite: bool) -> dict:
        """
        A preprocessing pipeline method to convert raw DLC data into preprocessed
        DLC data that is ready for ML analysis.
        All functs passed in must have the format func(df, dict) -> df. Possible funcs
        are given in preprocessing.py
        The preprocessed data is saved to the project's preprocessed folder.

        Parameters
        ----------
        funcs : tuple[Callable, ...]
            _description_
        overwrite : bool
            Whether to overwrite the output file (if it exists).

        Returns
        -------
        dict
            Diagnostics dictionary, with description of each function's outcome.

        Notes
        -----
        Can call any methods from `Preprocess`.
        """
        # Exporting 3_dlc df to 4_preprocessed folder
        dd = self._process_scaffold(
            (Export.feather2feather,),
            src_fp=self.get_fp(Folders.DLC),
            out_fp=self.get_fp(Folders.PREPROCESSED),
            overwrite=overwrite,
        )
        # If there is an error, OR warning (indicates not to ovewrite), then return early
        res = dd[Export.feather2feather.__name__]
        if res.startswith("ERROR") or res.startswith("WARNING"):
            return dd
        # Feeding through preprocessing functions
        return self._process_scaffold(
            funcs,
            dlc_fp=self.get_fp(Folders.PREPROCESSED),
            out_fp=self.get_fp(Folders.PREPROCESSED),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=True,
        )

    #####################################################################
    #                 SIMBA BEHAVIOUR CLASSIFICATION METHODS
    #####################################################################

    def extract_features(self, overwrite: bool) -> dict:
        """
        Extracts features from the preprocessed dlc file to generate many more features.
        This dataframe of derived features will be input for a ML classifier to detect
        particularly trained behaviours.

        Parameters
        ----------
        overwrite : bool
            Whether to overwrite the output file (if it exists).

        Returns
        -------
        dict
            Diagnostics dictionary, with description of each function's outcome.
        """
        return self._process_scaffold(
            (ExtractFeatures.extract_features,),
            dlc_fp=self.get_fp(Folders.PREPROCESSED),
            out_fp=self.get_fp(Folders.FEATURES_EXTRACTED),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    def classify_behaviours(self, overwrite: bool) -> dict:
        """
        Given model config files in the BehavClassifier format, generates beahviour predidctions
        on the given extracted features dataframe.

        Parameters
        ----------
        overwrite : bool
            Whether to overwrite the output file (if it exists).

        Returns
        -------
        dict
            Diagnostics dictionary, with description of each function's outcome.
        """
        return self._process_scaffold(
            (ClassifyBehavs.classify_behavs,),
            features_fp=self.get_fp(Folders.FEATURES_EXTRACTED),
            out_fp=self.get_fp(Folders.PREDICTED_BEHAVS),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    def export_behaviours(self, overwrite: bool) -> dict:
        """
        _summary_

        Parameters
        ----------
        overwrite : bool
            _description_

        Returns
        -------
        dict
            _description_
        """
        # Exporting 6_predicted_behavs df to 7_scored_behavs folder
        return self._process_scaffold(
            (Export.predbehavs2scoredbehavs,),
            src_fp=self.get_fp(Folders.PREDICTED_BEHAVS),
            out_fp=self.get_fp(Folders.SCORED_BEHAVS),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    #####################################################################
    #                     SIMPLE ANALYSIS METHODS
    #####################################################################

    def analyse(self, funcs: tuple[Callable, ...]) -> dict:
        """
        An ML pipeline method to analyse the preprocessed DLC data.
        Possible funcs are given in analysis.py.
        The preprocessed data is saved to the project's analysis folder.

        Parameters
        ----------
        funcs : tuple[Callable, ...]
            _description_

        Returns
        -------
        dict
            Diagnostics dictionary, with description of each function's outcome.

        Notes
        -----
        Can call any methods from `Analyse`.
        """
        return self._process_scaffold(
            funcs,
            dlc_fp=self.get_fp(Folders.PREPROCESSED),
            out_dir=os.path.join(self.root_dir, ANALYSIS_DIR),
            configs_fp=self.get_fp(Folders.CONFIGS),
        )

    def analyse_behaviours(self) -> dict:
        """
        An ML pipeline method to analyse the preprocessed DLC data.
        Possible funcs are given in analysis.py.
        The preprocessed data is saved to the project's analysis folder.

        Returns
        -------
        dict
            Diagnostics dictionary, with description of each function's outcome.

        Notes
        -----
        Can call any methods from `Analyse`.
        """
        return self._process_scaffold(
            (AnalyseBehaviours.analyse_behaviours,),
            behavs_fp=self.get_fp(Folders.SCORED_BEHAVS),
            out_dir=os.path.join(self.root_dir, ANALYSIS_DIR),
            configs_fp=self.get_fp(Folders.CONFIGS),
        )

    def analyse_combine(self, overwrite: bool) -> dict:
        """
        Combine the experiment's analysis in each fbf into a single df
        """
        # TODO: make new subfolder called combined_analysis and make ONLY(??) fbf analysis.
        return self._process_scaffold(
            (AnalyseCombine.analyse_combine,),
            analyse_dir=os.path.join(self.root_dir, ANALYSIS_DIR),
            out_fp=self.get_fp(Folders.ANALYSE_COMBINED),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    #####################################################################
    #           EVALUATING DLC ANALYSIS AND BEHAV CLASSIFICATION
    #####################################################################

    def evaluate_vid(self, overwrite: bool) -> dict:
        """
        Evaluating preprocessed DLC data and scored_behavs data.

        Parameters
        ----------
        funcs : _type_
            _description_
        overwrite : bool
            Whether to overwrite the output file (if it exists).

        Returns
        -------
        dict
            Diagnostics dictionary, with description of each function's outcome.
        """
        return self._process_scaffold(
            (EvaluateVid.evaluate_vid,),
            vid_fp=self.get_fp(Folders.FORMATTED_VID),
            dlc_fp=self.get_fp(Folders.PREPROCESSED),
            analyse_combined_fp=self.get_fp(Folders.ANALYSE_COMBINED),
            out_fp=self.get_fp(Folders.EVALUATE_VID),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    def export_feather(self, in_dir: str, out_dir: str, overwrite: bool) -> dict:
        """
        _summary_

        Parameters
        ----------
        in_dir : str
            _description_
        out_dir : str
            _description_

        Returns
        -------
        dict
            _description_
        """
        return self._process_scaffold(
            (Export.feather2csv,),
            in_fp=self.get_fp(in_dir),
            out_fp=os.path.join(out_dir, f"{self.name}.csv"),
            overwrite=overwrite,
        )
