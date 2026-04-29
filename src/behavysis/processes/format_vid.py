"""Functions have the following format:

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
from pathlib import Path

import cv2

from behavysis.models.experiment_configs import ExperimentConfigs
from behavysis.models.processes.format_vid import VidMetadata
from behavysis.utils.diagnostics_utils import file_exists_msg
from behavysis.utils.subproc_utils import run_subproc_console

logger = logging.getLogger(__name__)


class FormatVid:
    """Video formatting and metadata extraction for the pipeline.

    Provides methods to:
    - Format videos with ffmpeg (resize, trim, change fps)
    - Extract and store video metadata in configs
    """

    @classmethod
    def format_vid(
        cls,
        raw_vid_fp: Path,
        formatted_vid_fp: Path,
        configs_fp: Path,
        overwrite: bool,
    ) -> None:
        """Formats the input video with the given parameters.

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

        Returns:
        -------
        str
            Description of the function's outcome.
        """
        if not overwrite and formatted_vid_fp.exists():
            logger.warning(file_exists_msg(formatted_vid_fp))
            return
        # Finding all necessary config parameters for video formatting
        configs = ExperimentConfigs.model_validate_json(configs_fp.read_text())
        configs_filt = configs.user.format_vid
        # Processing the video
        ffmpeg_process_vid(
            in_fp=raw_vid_fp,
            dst_fp=formatted_vid_fp,
            width_px=configs.get_ref(configs_filt.width_px),
            height_px=configs.get_ref(configs_filt.height_px),
            fps=configs.get_ref(configs_filt.fps),
            start_sec=configs.get_ref(configs_filt.start_sec),
            stop_sec=configs.get_ref(configs_filt.stop_sec),
            overwrite=overwrite,
        )
        cls.get_vids_metadata(raw_vid_fp, formatted_vid_fp, configs_fp)

    @classmethod
    def get_vids_metadata(
        cls, raw_vid_fp: Path, formatted_vid_fp: Path, configs_fp: Path
    ) -> None:
        """Extract metadata from raw and formatted videos, store in configs.

        Parameters
        ----------
        raw_vid_fp : Path
            Raw video filepath.
        formatted_vid_fp : Path
            Formatted video filepath.
        configs_fp : Path
            JSON configs filepath to update with metadata.
        """
        # Saving video metadata to configs dict
        configs = ExperimentConfigs.model_validate_json(configs_fp.read_text())
        configs.auto.raw_vid = get_vid_metadata(raw_vid_fp)
        configs.auto.formatted_vid = get_vid_metadata(formatted_vid_fp)
        logger.info("Video metadata stored in config file.")
        configs_fp.write_text(configs.model_dump_json(indent=2))


def ffmpeg_process_vid(
    in_fp: Path,
    dst_fp: Path,
    width_px: None | int = None,
    height_px: None | int = None,
    fps: None | int = None,
    start_sec: None | float = None,
    stop_sec: None | float = None,
    overwrite: bool = False,
) -> None:
    """Process video with ffmpeg to resize, trim, and change frame rate.

    Parameters
    ----------
    in_fp : Path
        Input video filepath.
    dst_fp : Path
        Output video filepath.
    width_px : int, optional
        Target width in pixels. -1 auto-calculates from height.
    height_px : int, optional
        Target height in pixels. -1 auto-calculates from width.
    fps : int, optional
        Target frame rate.
    start_sec : float, optional
        Start time in seconds for trimming.
    stop_sec : float, optional
        Stop time in seconds for trimming.
    overwrite : bool
        Whether to overwrite existing output file.
    """
    # Constructing ffmpeg command
    cmd = ["ffmpeg"]

    # TRIMMING (SEEKING TO START BEFORE OPENING VIDEO - MUCH FASTER)
    if start_sec:
        # Setting start trim filter in cmd
        cmd += ["-ss", str(start_sec)]
        logger.debug(f"Trimming video from {start_sec} seconds.")

    # Opening video
    cmd += ["-i", str(in_fp)]

    # RESIZING and TRIMMING
    filters = []
    if width_px or height_px:
        # Setting width and height (if one is None)
        width_px = width_px or -1
        height_px = height_px or -1
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
        str(dst_fp),
    ]
    dst_fp.parent.mkdir(parents=True, exist_ok=True)
    run_subproc_console(cmd)


def get_vid_metadata(vid_fp: Path) -> VidMetadata:
    """Finds the video metadata/parameters for either the raw or formatted video.

    Parameters
    ----------
    fp : Path
        The video filepath.

    Returns:
    -------
    VidMetadata
        Object containing video metadata.
    """
    configs_meta = VidMetadata()
    cap = cv2.VideoCapture(vid_fp)
    if not cap.isOpened():
        logger.warning(
            "The file, %s, does not exist or is corrupted. Please check this file.",
            vid_fp,
        )
    else:
        configs_meta.height_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        configs_meta.width_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        configs_meta.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        configs_meta.fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return configs_meta
