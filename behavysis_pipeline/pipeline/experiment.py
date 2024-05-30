"""
_summary_
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import numpy as np
from behavysis_core.constants import (
    ANALYSIS_DIR,
    EVALUATE_DIR,
    FILE_EXTS,
    STR_DIV,
    TEMP_DIR,
    Folders,
)
from behavysis_core.mixins.diagnostics_mixin import DiagnosticsMixin

from behavysis_pipeline.processes import (
    BehavAnalyse,
    ClassifyBehaviours,
    Export,
    ExtractFeatures,
    RunDLC,
    UpdateConfigs,
)


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

    def __init__(self, name: str, root_dir: str) -> None:
        """
        Make a Experiment instance.
        """
        # Assertion: root_dir mus† exist
        if not os.path.isdir(root_dir):
            raise ValueError(
                f'Cannot find the project folder named "{root_dir}".\n'
                + "Please specify a folder that exists."
            )
        # Assertion: name must correspond to at least one file in root_dir
        file_exists_ls = [
            os.path.isfile(os.path.join(root_dir, f.value, f"{name}{FILE_EXTS[f]}"))
            for f in Folders
        ]
        if not np.any(file_exists_ls):
            raise ValueError(
                f'No files named "{name}" exist in "{root_dir}".\n'
                + f'Please specify a file that exists in "{root_dir}", in one of the'
                + " following folder WITH the correct file extension name:\n"
                + "    - "
                + "\n    - ".join([i.value for i in Folders])
            )
        self.name = name
        self.root_dir = os.path.abspath(root_dir)

    #####################################################################
    #               GET/CHECK FILEPATH METHODS
    #####################################################################

    def get_fp(self, folder_str: str) -> str:
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
        # Getting folder enum from string
        folder = next((f for f in Folders if folder_str == f.value), None)
        # Assertion: The given folder name must be valid
        if not folder:
            raise ValueError(
                f'"{folder_str}" is not a valid experiment folder name.\n'
                + "Please only specify one of the following folders:\n"
                + "    - "
                + "\n    - ".join([f.value for f in Folders])
            )
        # Getting experiment filepath for given folder
        fp = os.path.join(
            self.root_dir, folder.value, f"{self.name}{FILE_EXTS[folder]}"
        )
        # Making a folder if it does not exist
        os.makedirs(os.path.split(fp)[0], exist_ok=True)
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
        logging.info(f"Processing experiment: {self.name}")
        # Setting up diagnostics dict
        dd = {"experiment": self.name}
        # Running functions and saving outcome to diagnostics dict
        for f in funcs:
            # Running each func and saving outcome
            try:
                dd[f.__name__] = f(*args, **kwargs)
                dd[f.__name__] += f"SUCCESS: {DiagnosticsMixin.success_msg()}\n"
            except Exception as e:
                dd[f.__name__] = f"ERROR: {e}"
            logging.info(f"{f.__name__}: {dd[f.__name__]}")
        logging.info(STR_DIV)
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
            configs_fp=self.get_fp(Folders.CONFIGS.value),
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
            in_fp=self.get_fp(Folders.RAW_VID.value),
            out_fp=self.get_fp(Folders.FORMATTED_VID.value),
            configs_fp=self.get_fp(Folders.CONFIGS.value),
            overwrite=overwrite,
        )

    #####################################################################
    #                        DLC METHODS
    #####################################################################

    def run_dlc(self, gputouse: int, overwrite: bool) -> dict:
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
            in_fp=self.get_fp(Folders.FORMATTED_VID.value),
            out_fp=self.get_fp(Folders.DLC.value),
            configs_fp=self.get_fp(Folders.CONFIGS.value),
            temp_dir=os.path.join(self.root_dir, TEMP_DIR),
            gputouse=gputouse,
            overwrite=overwrite,
        )

    def calculate_params(self, funcs: tuple[Callable, ...]) -> dict:
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
            dlc_fp=self.get_fp(Folders.DLC.value),
            configs_fp=self.get_fp(Folders.CONFIGS.value),
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
            (Export.feather_2_feather,),
            in_fp=self.get_fp(Folders.DLC.value),
            out_fp=self.get_fp(Folders.PREPROCESSED.value),
            overwrite=overwrite,
        )
        # If there is an error, OR warning (indicates not to ovewrite), then return early
        res = dd["feather_2_feather"]
        if res.startswith("ERROR") or res.startswith("WARNING"):
            return dd
        # Feeding through preprocessing functions
        return self._process_scaffold(
            funcs,
            in_fp=self.get_fp(Folders.PREPROCESSED.value),
            out_fp=self.get_fp(Folders.PREPROCESSED.value),
            configs_fp=self.get_fp(Folders.CONFIGS.value),
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
            dlc_fp=self.get_fp(Folders.PREPROCESSED.value),
            out_fp=self.get_fp(Folders.FEATURES_EXTRACTED.value),
            configs_fp=self.get_fp(Folders.CONFIGS.value),
            temp_dir=os.path.join(self.root_dir, TEMP_DIR),
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
            (ClassifyBehaviours.classify_behaviours,),
            features_fp=self.get_fp(Folders.FEATURES_EXTRACTED.value),
            out_fp=self.get_fp(Folders.PREDICTED_BEHAVS.value),
            configs_fp=self.get_fp(Folders.CONFIGS.value),
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
            (Export.behav_export,),
            in_fp=self.get_fp(Folders.PREDICTED_BEHAVS.value),
            out_fp=self.get_fp(Folders.SCORED_BEHAVS.value),
            configs_fp=self.get_fp(Folders.CONFIGS.value),
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
            dlc_fp=self.get_fp(Folders.PREPROCESSED.value),
            analysis_dir=os.path.join(self.root_dir, ANALYSIS_DIR),
            configs_fp=self.get_fp(Folders.CONFIGS.value),
        )

    def behav_analyse(self) -> dict:
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
            (BehavAnalyse.behav_analysis,),
            behavs_fp=self.get_fp(Folders.SCORED_BEHAVS.value),
            analysis_dir=os.path.join(self.root_dir, ANALYSIS_DIR),
            configs_fp=self.get_fp(Folders.CONFIGS.value),
        )

    #####################################################################
    #           EVALUATING DLC ANALYSIS AND BEHAV CLASSIFICATION
    #####################################################################

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
            (Export.feather_2_csv,),
            in_fp=self.get_fp(in_dir),
            out_fp=os.path.join(out_dir, f"{self.name}.csv"),
            overwrite=overwrite,
        )

    def evaluate(self, funcs, overwrite: bool) -> dict:
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
            funcs,
            vid_fp=self.get_fp(Folders.FORMATTED_VID.value),
            dlc_fp=self.get_fp(Folders.PREPROCESSED.value),
            behavs_fp=self.get_fp(Folders.SCORED_BEHAVS.value),
            out_dir=os.path.join(self.root_dir, EVALUATE_DIR),
            configs_fp=self.get_fp(Folders.CONFIGS.value),
            overwrite=overwrite,
        )
