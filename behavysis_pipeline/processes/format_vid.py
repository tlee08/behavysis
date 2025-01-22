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

import os

import cv2

from behavysis_pipeline.pydantic_models.configs import ExperimentConfigs
from behavysis_pipeline.pydantic_models.vid_metadata import VidMetadata
from behavysis_pipeline.utils.diagnostics_utils import file_exists_msg
from behavysis_pipeline.utils.logging_utils import init_logger_with_io_obj, io_obj_to_msg
from behavysis_pipeline.utils.misc_utils import get_current_funct_name
from behavysis_pipeline.utils.subproc_utils import run_subproc_console


class FormatVid:
    """
    Class for formatting videos based on given parameters.
    """

    @classmethod
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
        logger, io_obj = init_logger_with_io_obj(get_current_funct_name())
        if not overwrite and os.path.exists(out_fp):
            logger.warning(file_exists_msg(out_fp))
            return io_obj_to_msg(io_obj)
        # Finding all necessary config parameters for video formatting
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = configs.user.format_vid

        # Processing the video
        logger.info(
            ProcessVidMixin.process_vid(
                in_fp=in_fp,
                out_fp=out_fp,
                height_px=configs.get_ref(configs_filt.height_px),
                width_px=configs.get_ref(configs_filt.width_px),
                fps=configs.get_ref(configs_filt.fps),
                start_sec=configs.get_ref(configs_filt.start_sec),
                stop_sec=configs.get_ref(configs_filt.stop_sec),
            )
        )

        # Saving video metadata to configs dict
        logger.info(FormatVid.get_vid_metadata(in_fp, out_fp, configs_fp, overwrite))
        return io_obj_to_msg(io_obj)

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
        logger, io_obj = init_logger_with_io_obj(get_current_funct_name())
        # Saving video metadata to configs dict
        configs = ExperimentConfigs.read_json(configs_fp)
        for ftype, fp in (("raw_vid", in_fp), ("formatted_vid", out_fp)):
            try:
                setattr(configs.auto, ftype, ProcessVidMixin.get_vid_metadata(fp))
            except ValueError as e:
                logger.warning(str(e))
        logger.info("Video metadata stored in config file.")
        configs.write_json(configs_fp)
        return io_obj_to_msg(io_obj)


class ProcessVidMixin:
    """__summary__"""

    @classmethod
    def process_vid(
        cls,
        in_fp: str,
        out_fp: str,
        height_px: None | int = None,
        width_px: None | int = None,
        fps: None | int = None,
        start_sec: None | float = None,
        stop_sec: None | float = None,
    ) -> str:
        """__summary__"""
        logger, io_obj = init_logger_with_io_obj(get_current_funct_name())
        # Constructing ffmpeg command
        cmd = ["ffmpeg"]

        # TRIMMING (SEEKING TO START BEFORE OPENING VIDEO - MUCH FASTER)
        if start_sec:
            # Setting start trim filter in cmd
            cmd += ["-ss", str(start_sec)]
            logger.debug(f"Trimming video from {start_sec} seconds.")

        # Opening video
        cmd += ["-i", in_fp]

        # RESIZING and TRIMMING
        filters = []
        if width_px or height_px:
            # Setting width and height (if one is None)
            width_px = width_px if width_px else -1
            height_px = height_px if height_px else -1
            # Constructing downsample filter in cmd
            filters.append(f"scale={width_px}:{height_px}")
            logger.debug(f"Downsampling to {width_px} x {height_px}.")
        # if start_sec or stop_sec:
        #     # Preparing start-stop filter in cmd
        #     filters.append("setpts=PTS-STARTPTS")
        if filters:
            cmd += ["-vf", ",".join(filters)]

        # CHANGING FPS
        if fps:
            cmd += ["-r", str(fps)]
            logger.debug(f"Changing fps to {fps}.")
        # TRIMMING
        if stop_sec:
            # Setting stop trim filter in cmd
            duration = stop_sec - (start_sec or 0)
            cmd += ["-t", str(duration)]
            logger.debug(f"Trimming video to {stop_sec} seconds.")

        # Adding output parameters to ffmpeg command
        cmd += [
            "-c:v",
            "h264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-y",
            # "-loglevel",
            # "quiet",
            out_fp,
        ]
        # Making the output directory
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        # Running ffmpeg command
        # run_subproc_fstream(cmd)
        run_subproc_console(cmd)
        return io_obj_to_msg(io_obj)

    @classmethod
    def get_vid_metadata(cls, fp: str) -> VidMetadata:
        """
        Finds the video metadata/parameters for either the raw or formatted video.

        Parameters
        ----------
        fp : str
            The video filepath.

        Returns
        -------
        VidMetadata
            Object containing video metadata.
        """
        configs_meta = VidMetadata()
        cap = cv2.VideoCapture(fp)
        if not cap.isOpened():
            raise ValueError(f"The file, {fp}, does not exist or is corrupted. Please check this file.")
        configs_meta.height_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        configs_meta.width_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        configs_meta.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        configs_meta.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return configs_meta
