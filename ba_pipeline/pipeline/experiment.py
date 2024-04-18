"""
_summary_
"""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from ba_pipeline.pipeline.experiment_configs import ExperimentConfigs
from ba_pipeline.processes import (
    ClassifyBehaviours,
    ExtractFeatures,
    RunDLC,
    UpdateConfigs,
)
from ba_pipeline.utils.constants import (
    ANALYSIS_DIR,
    EVALUATE_DIR,
    FOLDERS,
    STR_DIV,
    TEMP_DIR,
)
from ba_pipeline.utils.funcs import (
    read_feather,
    success_msg,
    warning_msg,
    write_feather,
)


class BAExperiment:
    """
    Behavioral Analysis Pipeline class for a single experiment.

    Encompasses the entire process including:
    - Raw mp4 file import.
    - mp4 file formatting (px and fps).
    - DLC keypoints inference.
    - Feature wrangling (start time detection, more features like average body position).
    - Interpretable behaviour analysis results.
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
        Make a BAExperiment instance.
        """
        # Assertion: root_dir mus† exist
        if not os.path.isdir(root_dir):
            raise ValueError(
                f'Cannot find the project folder named "{root_dir}".\n'
                + "Please specify a folder that exists."
            )
        # Assertion: name must correspond to at least one file in root_dir
        folder_fp_ls = [
            os.path.join(root_dir, folder, f"{name}{ext}")
            for folder, ext in FOLDERS.items()
        ]
        if not np.any([os.path.isfile(i) for i in folder_fp_ls]):
            raise ValueError(
                f'No files named "{name}" exist in "{root_dir}".\n'
                + f'Please specify a file that exists in "{root_dir}", in one of the'
                + " following folder WITH the correct file extension name:\n"
                + "    - "
                + "\n    - ".join(FOLDERS.keys())
            )
        self.name = name
        self.root_dir = os.path.abspath(root_dir)

    #####################################################################
    #               GET/CHECK FILEPATH METHODS
    #####################################################################

    def get_fp(self, folder: str) -> str:
        """
        Returns the experiment's file path from the given folder.

        Parameters
        ----------
        folder : str
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
        # Assertion: The given folder name must be valid
        if folder not in FOLDERS:
            raise ValueError(
                f'"{folder}" is not a valid experiment folder name.\n'
                + "Please only specify one of the following folders:\n"
                + "    - "
                + "\n    - ".join(FOLDERS.keys())
            )
        # Getting experiment filepath for given folder
        fp = os.path.join(self.root_dir, folder, f"{self.name}{FOLDERS[folder]}")
        # Making a folder if it does not exist
        os.makedirs(os.path.split(fp)[0], exist_ok=True)
        # Returning filepath
        return fp

    def check_fp(self, folder: str) -> bool:
        """
        Returns whether the corresponding experiment file exists in the given folder.

        Parameters
        ----------
        folder : str
            The folder to check the experiment document's filepath for.

        Returns
        -------
        bool
            Boolean outcome of whether the corresponding file exists.
        """
        fp = self.get_fp(folder)
        return os.path.isfile(fp)

    #####################################################################
    #               EXPERIMENT PROCESSING SCAFFOLD METHODS
    #####################################################################

    def _process_scaffold(
        self,
        funcs: tuple[Callable, ...],
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        """
        All processing runs through here.
        This method ensures that the stdout and diagnostics dict are correctly generated.

        Parameters
        ----------
        funcs : tuple[Callable, ...]
            List of functions.

        Returns
        -------
        dict
            Diagnostics dictionary, with description of each function's outcome.

        Notes
        -----
        Each func in `funcs` is called in the form:
        ```
        func(*args, **kwargs)
        ```
        """
        # Setting up diagnostics dict
        dd = {"experiment": self.name}
        print(f"Processing experiment: {self.name}")
        # Running functions and saving outcome to diagnostics dict
        for f in funcs:
            # Running each func and saving outcome
            try:
                dd[f.__name__] = f(*args, **kwargs)
                dd[f.__name__] += f"SUCCESS: {success_msg()}\n"
            except Exception as e:
                dd[f.__name__] = f"ERROR: {e}"
            # Printing outcome
            print(f"{f.__name__}: {dd[f.__name__]}")
        print(STR_DIV)
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
            configs_fp=self.get_fp("0_configs"),
            default_configs_fp=default_configs_fp,
            overwrite=overwrite,
            model_class=ExperimentConfigs,
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
        Can call any functions from inside `FormatVid`
        """
        return self._process_scaffold(
            funcs,
            raw_vid_fp=self.get_fp("1_raw_vid"),
            formatted_vid_fp=self.get_fp("2_formatted_vid"),
            configs_fp=self.get_fp("0_configs"),
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
        Can call any functions from inside `RunDLC`
        """
        return self._process_scaffold(
            (RunDLC.ma_dlc_analyse,),
            in_fp=self.get_fp("2_formatted_vid"),
            out_fp=self.get_fp("3_dlc"),
            configs_fp=self.get_fp("0_configs"),
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
        Can call any functions from inside `CalculateParams`
        """
        return self._process_scaffold(
            funcs,
            dlc_fp=self.get_fp("3_dlc"),
            configs_fp=self.get_fp("0_configs"),
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
        Can call any functions from inside `Preprocess`
        """
        # Exporting 3_dlc df to 4_preprocessed folder
        # If there is an error, then makes the diagnostics dict
        # where all function outcomes have the error message
        try:
            df = read_feather(self.get_fp("3_dlc"))
            write_feather(df, self.get_fp("4_preprocessed"))
        except Exception as e:
            dd = {f.__name__: str(e) for f in funcs}
            dd["experiment"] = self.name
            return dd

        # Feeding through preprocessing functions
        return self._process_scaffold(
            funcs,
            in_fp=self.get_fp("4_preprocessed"),
            out_fp=self.get_fp("4_preprocessed"),
            configs_fp=self.get_fp("0_configs"),
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
        Can call any functions from inside `Analyse`
        """
        return self._process_scaffold(
            funcs,
            dlc_fp=self.get_fp("4_preprocessed"),
            analysis_dir=os.path.join(self.root_dir, ANALYSIS_DIR),
            configs_fp=self.get_fp("0_configs"),
        )

    def aggregate_analysis(self, overwrite: bool) -> dict:
        """
        Combines all the frame-by-frame (fbf) analysis measures for the single experiment.
        The index is (frame, timestamp) and the columns are (analysisFile, indiv, measure)

        Parameters
        ----------
        overwrite : bool
            Whether to overwrite the output file (if it exists).

        Returns
        -------
        dict
            Diagnostics dictionary, with description of each function's outcome.
        """
        dd = {"experiment": self.name}
        analysis_dir = os.path.join(self.root_dir, ANALYSIS_DIR)
        out_fp = os.path.join(
            analysis_dir, "aggregateAnalysis", "fbf", f"{self.name}.feather"
        )
        # If overwrite is False, checking if we should skip processing
        if not overwrite and os.path.exists(out_fp):
            dd["aggregate_analysis"] = warning_msg()
            return dd
        # Making total_df to store all frame-by-frame analysis for the experiment
        total_df = pd.DataFrame()
        # Searching through all the analysis sub-folders
        for analysis_subdir in os.listdir(analysis_dir):
            if analysis_subdir == "aggregate_analysis":
                continue
            in_fp = os.path.join(
                analysis_dir, analysis_subdir, "fbf", f"{self.name}.feather"
            )
            if os.path.isfile(in_fp):
                # Reading exp fbf df
                df = read_feather(in_fp)
                # Asserting that the index is the same (frames, timestamps)
                if total_df.shape[0] > 0:
                    assert np.all(total_df.index == df.index)
                # Prepending analysis level to columns MultiIndex
                df = pd.concat([df], keys=[analysis_subdir], names=["analysis"], axis=1)
                # Concatenating total_df with df
                total_df = pd.concat([total_df, df], axis=1)
        write_feather(total_df, out_fp)
        return dd

    #####################################################################
    #                 SIMBA BEHAVIOUR CLASSIFICATION METHODS
    #####################################################################

    def extract_features(self, remove_temp: bool, overwrite: bool) -> dict:
        """
        Extracts features from the preprocessed dlc file to generate many more features.
        This dataframe of derived features will be input for a ML classifier to detect
        particularly trained behaviours.

        Parameters
        ----------
        remove_temp : bool
            Whether to remove the temp directory.
        overwrite : bool
            Whether to overwrite the output file (if it exists).

        Returns
        -------
        dict
            Diagnostics dictionary, with description of each function's outcome.
        """
        return self._process_scaffold(
            (ExtractFeatures.extract_features,),
            dlc_fp=self.get_fp("4_preprocessed"),
            out_fp=self.get_fp("5_features_extracted"),
            configs_fp=self.get_fp("0_configs"),
            temp_dir=os.path.join(self.root_dir, TEMP_DIR),
            remove_temp=remove_temp,
            overwrite=overwrite,
        )

    def classify_behaviours(self, overwrite: bool) -> dict:
        """
        Given model config files in the SimbaClassifier format, generates beahviour predidctions
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
            features_fp=self.get_fp("5_features_extracted"),
            out_fp=self.get_fp("6_predicted_behavs"),
            configs_fp=self.get_fp("0_configs"),
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
        dd = {"experiment": self.name}
        if not overwrite and os.path.exists(self.get_fp("7_scored_behavs")):
            dd["export_behaviours"] = warning_msg()
            return dd
        shutil.copyfile(
            self.get_fp("6_predicted_behavs"),
            self.get_fp("7_scored_behavs"),
        )
        dd["export"] = (
            "Copied predicted_behavs dataframe to 7_scored_behavs folder. "
            + "Ready for ba_viewer scoring!"
        )
        return dd

    #####################################################################
    #           EVALUATING DLC ANALYSIS AND BEHAV CLASSIFICATION
    #####################################################################

    def export_feather(self, in_dir: str, out_dir: str) -> dict:
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
        df = read_feather(self.get_fp(in_dir))
        df.to_csv(os.path.join(out_dir, f"{self.name}.csv"))
        return {
            "experiment": self.name,
            "export": f"Copied {in_dir} dataframe to {out_dir} folder. "
            + "Ready to view!",
        }

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
            vid_fp=self.get_fp("2_formatted_vid"),
            dlc_fp=self.get_fp("4_preprocessed"),
            behav_fp=self.get_fp("6_predicted_behavs"),
            out_dir=os.path.join(self.root_dir, EVALUATE_DIR),
            configs_fp=self.get_fp("0_configs"),
            overwrite=overwrite,
        )
