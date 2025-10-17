"""
Functions have the following format:

Parameters
----------
raw_fp : str
    The input video filepath.
formatted_fp : str
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

import logging
import os

import cv2

from behavysis.models.experiment_configs import ExperimentConfigs
from behavysis.models.processes.format_vid import VidMetadata
from behavysis.utils.diagnostics_utils import file_exists_msg
from behavysis.utils.logging_utils import (
    get_io_obj_content,
    init_logger_console,
    init_logger_io_obj,
)
from behavysis.utils.subproc_utils import run_subproc_console

# TODO: Maybe separate format_vid and get_vids_metadata into separate classes and processes


class FormatVid:
    """
    Class for formatting videos based on given parameters.
    """

    @classmethod
    def format_vid(cls, raw_vid_fp: str, formatted_vid_fp: str, configs_fp: str, overwrite: bool) -> str:
        """
        Formats the input video with the given parameters.

        Parameters
        ----------
        raw_fp : str
            The input video filepath.
        formatted_fp : str
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
        logger, io_obj = init_logger_io_obj()
        if not overwrite and os.path.exists(formatted_vid_fp):
            logger.warning(file_exists_msg(formatted_vid_fp))
            return get_io_obj_content(io_obj)
        # Finding all necessary config parameters for video formatting
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = configs.user.format_vid
        # Processing the video
        ffmpeg_process_vid(
            in_fp=raw_vid_fp,
            dst_fp=formatted_vid_fp,
            logger=logger,
            width_px=configs.get_ref(configs_filt.width_px),
            height_px=configs.get_ref(configs_filt.height_px),
            fps=configs.get_ref(configs_filt.fps),
            start_sec=configs.get_ref(configs_filt.start_sec),
            stop_sec=configs.get_ref(configs_filt.stop_sec),
            overwrite=overwrite,
        )
        cls.get_vids_metadata(raw_vid_fp, formatted_vid_fp, configs_fp)
        return get_io_obj_content(io_obj)

    @classmethod
    def get_vids_metadata(cls, raw_vid_fp: str, formatted_vid_fp: str, configs_fp: str) -> str:
        """
        Finds the video metadata/parameters for either the raw or formatted video,
        and stores this data in the experiment's config file.

        Parameters
        ----------
        raw_fp : str
            The input video filepath.
        formatted_fp : str
            The output video filepath.
        configs_fp : str
            The JSON configs filepath.

        Returns
        -------
        str
            Description of the function's outcome.
        """
        logger, io_obj = init_logger_io_obj()
        # Saving video metadata to configs dict
        configs = ExperimentConfigs.read_json(configs_fp)
        configs.auto.raw_vid = get_vid_metadata(raw_vid_fp, logger)
        configs.auto.formatted_vid = get_vid_metadata(formatted_vid_fp, logger)
        logger.info("Video metadata stored in config file.")
        configs.write_json(configs_fp)
        return get_io_obj_content(io_obj)


def ffmpeg_process_vid(
    in_fp: str,
    dst_fp: str,
    logger: None | logging.Logger = None,
    width_px: None | int = None,
    height_px: None | int = None,
    fps: None | int = None,
    start_sec: None | float = None,
    stop_sec: None | float = None,
    overwrite: bool = False,
) -> None:
    """__summary__"""
    if not logger:
        logger = init_logger_console()
    if not overwrite and os.path.exists(dst_fp):
        logger.warning(file_exists_msg(dst_fp))
        return
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
        dst_fp,
    ]
    os.makedirs(os.path.dirname(dst_fp), exist_ok=True)
    run_subproc_console(cmd)


def get_vid_metadata(vid_fp: str, logger: logging.Logger) -> VidMetadata:
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
    cap = cv2.VideoCapture(vid_fp)
    if not cap.isOpened():
        logger.warning(f"The file, {vid_fp}, does not exist or is corrupted. Please check this file.")
    else:
        configs_meta.height_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        configs_meta.width_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        configs_meta.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        configs_meta.fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return configs_meta
