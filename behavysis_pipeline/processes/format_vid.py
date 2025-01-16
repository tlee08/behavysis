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

from behavysis_pipeline.df_classes.diagnostics_df import DiagnosticsMixin
from behavysis_pipeline.mixins.process_vid_mixin import ProcessVidMixin
from behavysis_pipeline.pydantic_models.experiment_configs import ExperimentConfigs
from behavysis_pipeline.utils.io_utils import IOMixin
from behavysis_pipeline.utils.logging_utils import func_decorator, init_logger


class FormatVid:
    """
    Class for formatting videos based on given parameters.
    """

    logger = init_logger(__name__)

    @classmethod
    @func_decorator(logger)
    def format_vid(cls, in_fp: str, out_fp: str, configs_fp: str, overwrite: bool) -> str:
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
        if not overwrite and IOMixin.check_files_exist(out_fp):
            return DiagnosticsMixin.file_exists_msg(out_fp)
        outcome = ""
        # Finding all necessary config parameters for video formatting
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = configs.user.format_vid

        # Processing the video
        outcome += ProcessVidMixin.process_vid(
            in_fp=in_fp,
            out_fp=out_fp,
            height_px=configs.get_ref(configs_filt.height_px),
            width_px=configs.get_ref(configs_filt.width_px),
            fps=configs.get_ref(configs_filt.fps),
            start_sec=configs.get_ref(configs_filt.start_sec),
            stop_sec=configs.get_ref(configs_filt.stop_sec),
        )

        # Saving video metadata to configs dict
        outcome += FormatVid.get_vid_metadata(in_fp, out_fp, configs_fp, overwrite)
        return outcome

    @classmethod
    def get_vid_metadata(cls, in_fp: str, out_fp: str, configs_fp: str, overwrite: bool) -> str:
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
            Whether to overwrite the output file (if it exists). IGNORED

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
