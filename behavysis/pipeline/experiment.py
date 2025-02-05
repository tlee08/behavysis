"""
_summary_
"""

import os
import traceback
from typing import Any, Callable, Dict

import numpy as np

from behavysis.constants import (
    ANALYSIS_DIR,
    FileExts,
    Folders,
)
from behavysis.processes.analyse_behavs import AnalyseBehavs
from behavysis.processes.classify_behavs import ClassifyBehavs
from behavysis.processes.combine_analysis import CombineAnalysis
from behavysis.processes.evaluate_vid import EvaluateVid
from behavysis.processes.export import Export
from behavysis.processes.extract_features import ExtractFeatures
from behavysis.processes.format_vid import FormatVid
from behavysis.processes.run_dlc import RunDLC
from behavysis.processes.update_configs import UpdateConfigs
from behavysis.pydantic_models.experiment_configs import AutoConfigs, ExperimentConfigs
from behavysis.utils.logging_utils import get_io_obj_content, init_logger_file, init_logger_io_obj


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

    logger = init_logger_file()

    def __init__(self, name: str, root_dir: str) -> None:
        """
        Make a Experiment instance.
        """
        # Assertion: root_dir mus† exist
        if not os.path.isdir(root_dir):
            raise ValueError(
                f'Cannot find the project folder named "{root_dir}".\nPlease specify a folder that exists.'
            )
        # Setting up instance variables
        self.name = name
        self.root_dir = os.path.abspath(root_dir)
        # Assertion: name must correspond to at least one file in root_dir
        file_exists_ls = [os.path.isfile(self.get_fp(f)) for f in Folders]
        if not np.any(file_exists_ls):
            folders_ls_msg = "".join([f"\n    - {f.value}" for f in Folders])
            raise ValueError(
                f'No files named "{name}" exist in "{root_dir}".\n'
                f'Please specify a file that exists in "{root_dir}", '
                f"in one of the following folder WITH the correct file extension name:{folders_ls_msg}"
            )

    #####################################################################
    #               GET/CHECK FILEPATH METHODS
    #####################################################################

    def get_fp(self, _folder: Folders | str) -> str:
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
            ValueError: Folder name is not valid. Refer to Folders Enum for valid folder names.
        """
        # Getting Folder item
        if isinstance(_folder, str):
            try:
                folder = Folders(_folder)
            except ValueError:
                folders_ls_msg = "".join([f"\n    - {f.value}" for f in Folders])
                raise ValueError(
                    f"{_folder} is not a valid experiment folder name.\n"
                    f"Please only specify one of the following folders:{folders_ls_msg}"
                )
        else:
            folder = _folder
        # Getting file extension from enum
        file_ext: FileExts = getattr(FileExts, folder.name)
        # Getting experiment filepath for given folder
        fp = os.path.join(self.root_dir, folder.value, f"{self.name}.{file_ext.value}")
        return fp

    #####################################################################
    #               EXPERIMENT PROCESSING SCAFFOLD METHODS
    #####################################################################

    def _proc_scaff(
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
        f_names_ls_msg = "".join([f"\n    - {f.__name__}" for f in funcs])
        self.logger.info(f"Processing experiment, {self.name}, with:{f_names_ls_msg}")
        # Setting up diagnostics dict
        dd = {"experiment": self.name}
        # Running functions and saving outcome to diagnostics dict
        for f in funcs:
            f_name = f.__name__
            # Getting logger and corresponding io object
            f_logger, f_io_obj = init_logger_io_obj(f_name)
            # Running each func and saving outcome
            try:
                f(*args, **kwargs)
                # f_logger.info(success_msg())
            except Exception as e:
                f_logger.error(e)
                self.logger.debug(traceback.format_exc())
            # Adding to diagnostics dict
            dd[f_name] = get_io_obj_content(f_io_obj)
            # Clearing io object
            f_io_obj.truncate(0)
        self.logger.info(f"Finished processing experiment, {self.name}, with:{f_names_ls_msg}")
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
        return self._proc_scaff(
            (UpdateConfigs.update_configs,),
            configs_fp=self.get_fp(Folders.CONFIGS),
            default_configs_fp=default_configs_fp,
            overwrite=overwrite,
        )

    #####################################################################
    #                    FORMATTING VIDEO METHODS
    #####################################################################

    def format_vid(self, overwrite: bool) -> dict:
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
        return self._proc_scaff(
            (FormatVid.format_vid,),
            raw_vid_fp=self.get_fp(Folders.RAW_VID),
            formatted_vid_fp=self.get_fp(Folders.FORMATTED_VID),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    def get_vid_metadata(self) -> dict:
        """
        Gets the video metadata for the raw and formatted video files.

        Parameters
        ----------
        overwrite : bool
            _description_

        Returns
        -------
        dict
            _description_
        """
        return self._proc_scaff(
            (FormatVid.get_vids_metadata,),
            raw_vid_fp=self.get_fp(Folders.RAW_VID),
            formatted_vid_fp=self.get_fp(Folders.FORMATTED_VID),
            configs_fp=self.get_fp(Folders.CONFIGS),
        )

    #####################################################################
    #                      DLC KEYPOINTS METHODS
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
        return self._proc_scaff(
            (RunDLC.ma_dlc_run_single,),
            formatted_vid_fp=self.get_fp(Folders.FORMATTED_VID),
            keypoints_fp=self.get_fp(Folders.KEYPOINTS),
            configs_fp=self.get_fp(Folders.CONFIGS),
            gputouse=gputouse,
            overwrite=overwrite,
        )

    def calculate_parameters(self, funcs: tuple[Callable, ...]) -> dict:
        """
        A pipeline to calculate the parameters of the keypoints file, which will
        assist in preprocessing the keypoints data.

        Parameters
        ----------
        funcs : tuple[Callable, ...]
            _description_

        Returns
        -------
        Dict
            Diagnostics dictionary, with description of each function's outcome.

        Notes
        -----
        Can call any methods from `CalculateParams`.
        """
        return self._proc_scaff(
            funcs,
            keypoints_fp=self.get_fp(Folders.KEYPOINTS),
            configs_fp=self.get_fp(Folders.CONFIGS),
        )

    def collate_auto_configs(self) -> Dict:
        """
        Collates the auto-configs of the experiment into the main configs file.
        """
        dd = {"experiment": self.name}
        # Reading the experiment's configs file
        f_logger, f_io_obj = init_logger_io_obj()
        try:
            configs = ExperimentConfigs.read_json(self.get_fp(Folders.CONFIGS))
            f_logger.debug("Reading configs file.")
            # f_logger.info(success_msg())
            dd["reading_configs"] = get_io_obj_content(f_io_obj)
        except FileNotFoundError:
            f_logger.error("no configs file found.")
            dd["reading_configs"] = get_io_obj_content(f_io_obj)
            return dd
        # Getting all the auto fields from the configs file
        configs_auto_field_keys = AutoConfigs.get_field_names()
        for field_key_ls in configs_auto_field_keys:
            value = configs.auto
            for key in field_key_ls:
                value = getattr(value, key)
            dd["_".join(field_key_ls)] = value  # type: ignore
        return dd

    def preprocess(self, funcs: tuple[Callable, ...], overwrite: bool) -> dict:
        """
        A preprocessing pipeline method to convert raw keypoints data into preprocessed
        keypoints data that is ready for ML analysis.
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
        # Exporting keypoints df to preprocessed folder
        dd0 = self._proc_scaff(
            (Export.df2df,),
            src_fp=self.get_fp(Folders.KEYPOINTS),
            dst_fp=self.get_fp(Folders.PREPROCESSED),
            overwrite=overwrite,
        )
        # If there is an error or warning (indicates not to ovewrite) in logger, return early
        if "ERROR" in dd0[Export.df2df.__name__] or "WARNING" in dd0[Export.df2df.__name__]:
            return dd0
        # Feeding through preprocessing functions
        dd1 = self._proc_scaff(
            funcs,
            src_fp=self.get_fp(Folders.PREPROCESSED),
            dst_fp=self.get_fp(Folders.PREPROCESSED),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=True,
        )
        return {**dd0, **dd1}

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
        return self._proc_scaff(
            (ExtractFeatures.extract_features,),
            keypoints_fp=self.get_fp(Folders.PREPROCESSED),
            features_fp=self.get_fp(Folders.FEATURES_EXTRACTED),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    def classify_behavs(self, overwrite: bool) -> dict:
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
        return self._proc_scaff(
            (ClassifyBehavs.classify_behavs,),
            features_fp=self.get_fp(Folders.FEATURES_EXTRACTED),
            behavs_fp=self.get_fp(Folders.PREDICTED_BEHAVS),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    def export_behavs(self, overwrite: bool) -> dict:
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
        return self._proc_scaff(
            (Export.predictedbehavs2scoredbehavs,),
            src_fp=self.get_fp(Folders.PREDICTED_BEHAVS),
            dst_fp=self.get_fp(Folders.SCORED_BEHAVS),
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
        return self._proc_scaff(
            funcs,
            keypoints_fp=self.get_fp(Folders.PREPROCESSED),
            dst_dir=os.path.join(self.root_dir, ANALYSIS_DIR),
            configs_fp=self.get_fp(Folders.CONFIGS),
        )

    def analyse_behavs(self) -> dict:
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
        return self._proc_scaff(
            (AnalyseBehavs.analyse_behavs,),
            behavs_fp=self.get_fp(Folders.SCORED_BEHAVS),
            dst_dir=os.path.join(self.root_dir, ANALYSIS_DIR),
            configs_fp=self.get_fp(Folders.CONFIGS),
        )

    def combine_analysis(self) -> dict:
        """
        Combine the experiment's analysis in each fbf into a single df
        """
        # TODO: make new subfolder called combined_analysis and make ONLY(??) fbf analysis.
        return self._proc_scaff(
            (CombineAnalysis.combine_analysis,),
            analysis_dir=os.path.join(self.root_dir, ANALYSIS_DIR),
            analysis_combined_fp=self.get_fp(Folders.ANALYSIS_COMBINED),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=True,  # TODO: remove overwrite
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
        return self._proc_scaff(
            (EvaluateVid.evaluate_vid,),
            formatted_vid_fp=self.get_fp(Folders.FORMATTED_VID),
            keypoints_fp=self.get_fp(Folders.PREPROCESSED),
            analysis_combined_fp=self.get_fp(Folders.ANALYSIS_COMBINED),
            eval_vid_fp=self.get_fp(Folders.EVALUATE_VID),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )

    def export2csv(self, src_dir: str, dst_dir: str, overwrite: bool) -> dict:
        """
        _summary_

        Parameters
        ----------
        src_dir : str
            _description_
        dst_dir : str
            _description_

        Returns
        -------
        dict
            _description_
        """
        return self._proc_scaff(
            (Export.df2csv,),
            src_fp=self.get_fp(src_dir),
            dst_fp=os.path.join(dst_dir, f"{self.name}.csv"),
            overwrite=overwrite,
        )
