"""
Functions have the following format:

Parameters
----------
raw_vid_fp : str
    The input video filepath.
formatted_vid_fp : str
    The output video filepath.
configs_fp : str
    The JSON configs filepath.
overwrite : bool
    Whether to overwrite the output file (if it exists).

Returnss
-------
str
    Description of the function's outcome.
"""

import os

from ba_core.data_models.experiment_configs import ExperimentConfigs
from ba_core.mixins.diagnostics_mixin import DiagnosticsMixin
from ba_core.mixins.process_vid_mixin import ProcessVidMixin


class FormatVid:
    """
    Class for formatting videos based on given parameters.
    """

    @staticmethod
    def format_vid(
        raw_vid_fp: str, formatted_vid_fp: str, configs_fp: str, overwrite: bool
    ) -> str:
        """
        Formats the input video with the given parameters.

        Parameters
        ----------
        raw_vid_fp : str
            The input video filepath.
        formatted_vid_fp : str
            The output video filepath.
        configs_fp : str
            The JSON configs filepath.
        overwrite : bool
            Whether to overwrite the output file (if it exists).

        Returns
        -------
        str
            Description of the function's outcome.
        """
        outcome = ""

        # If overwrite is False, checking if we should skip processing
        if not overwrite and os.path.exists(formatted_vid_fp):
            return DiagnosticsMixin.warning_msg()

        # Finding all necessary config parameters for video formatting
        configs = ExperimentConfigs.read_configs(configs_fp)
        configs_filt = configs.user.format_vid
        width_px = configs_filt.width_px
        height_px = configs_filt.height_px
        fps = configs_filt.fps
        start_sec = configs_filt.start_sec
        stop_sec = configs_filt.stop_sec

        # Processing the video
        outcome += ProcessVidMixin.process_vid(
            raw_vid_fp, formatted_vid_fp, height_px, width_px, fps, start_sec, stop_sec
        )

        # Saving video metadata to configs dict
        outcome += FormatVid.get_vid_metadata(
            raw_vid_fp, formatted_vid_fp, configs_fp, overwrite
        )
        return outcome

    @staticmethod
    def get_vid_metadata(
        raw_vid_fp: str, formatted_vid_fp: str, configs_fp: str, overwrite: bool
    ) -> str:
        """
        Finds the video metadata/parameters for either the raw or formatted video,
        and stores this data in the experiment's config file.

        Parameters
        ----------
        raw_vid_fp : str
            The input video filepath.
        formatted_vid_fp : str
            The output video filepath.
        configs_fp : str
            The JSON configs filepath.
        overwrite : bool
            Whether to overwrite the output file (if it exists).

        Returns
        -------
        str
            Description of the function's outcome.
        """
        outcome = ""

        # Saving video metadata to configs dict
        configs = ExperimentConfigs.read_configs(configs_fp)
        for ftype, fp in (("raw_vid", raw_vid_fp), ("formatted_vid", formatted_vid_fp)):
            try:
                setattr(configs.auto, ftype, ProcessVidMixin.get_vid_metadata(fp))
            except ValueError as e:
                outcome += f"WARNING: {str(e)}\n"
        outcome += "Video metadata stored in config file.\n"
        configs.write_configs(configs_fp)
        return outcome
