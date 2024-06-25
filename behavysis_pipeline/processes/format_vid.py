"""
Functions have the following format:

Parameters
----------
in_fp : str
    The input video filepath.
out_fp : str
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

from behavysis_core.data_models.experiment_configs import ExperimentConfigs
from behavysis_core.mixins.io_mixin import IOMixin
from behavysis_core.mixins.process_vid_mixin import ProcessVidMixin


class FormatVid:
    """
    Class for formatting videos based on given parameters.
    """

    @staticmethod
    @IOMixin.overwrite_check()
    def format_vid(in_fp: str, out_fp: str, configs_fp: str, overwrite: bool) -> str:
        """
        Formats the input video with the given parameters.

        Parameters
        ----------
        in_fp : str
            The input video filepath.
        out_fp : str
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
        # Finding all necessary config parameters for video formatting
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = configs.user.format_vid

        # Processing the video
        outcome += ProcessVidMixin.process_vid(
            in_fp=in_fp,
            out_fp=out_fp,
            height_px=configs_filt.height_px,
            width_px=configs_filt.width_px,
            fps=configs_filt.fps,
            start_sec=configs_filt.start_sec,
            stop_sec=configs_filt.stop_sec,
        )

        # Saving video metadata to configs dict
        outcome += FormatVid.get_vid_metadata(in_fp, out_fp, configs_fp, overwrite)
        return outcome

    @staticmethod
    @IOMixin.overwrite_check()
    def get_vid_metadata(
        in_fp: str, out_fp: str, configs_fp: str, overwrite: bool
    ) -> str:
        """
        Finds the video metadata/parameters for either the raw or formatted video,
        and stores this data in the experiment's config file.

        Parameters
        ----------
        in_fp : str
            The input video filepath.
        out_fp : str
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
        configs = ExperimentConfigs.read_json(configs_fp)
        for ftype, fp in (("raw_vid", in_fp), ("formatted_vid", out_fp)):
            try:
                setattr(configs.auto, ftype, ProcessVidMixin.get_vid_metadata(fp))
            except ValueError as e:
                outcome += f"WARNING: {str(e)}\n"
        outcome += "Video metadata stored in config file.\n"
        configs.write_json(configs_fp)
        return outcome
