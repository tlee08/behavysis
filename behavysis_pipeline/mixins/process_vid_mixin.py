"""
Utility functions.
"""

from __future__ import annotations

import os

import cv2

from behavysis_pipeline.pydantic_models.vid_metadata import VidMetadata
from behavysis_pipeline.utils.logging_utils import init_logger
from behavysis_pipeline.utils.subproc_utils import SubprocMixin


class ProcessVidMixin:
    """__summary__"""

    logger = init_logger(__name__)

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
        outcome = ""
        # Constructing ffmpeg command
        cmd = ["ffmpeg"]

        # TRIMMING (SEEKING TO START BEFORE OPENING VIDEO - MUCH FASTER)
        if start_sec:
            # Setting start trim filter in cmd
            cmd += ["-ss", str(start_sec)]
            outcome += f"Trimming video from {start_sec} seconds.\n"

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
            # Adding to outcome
            outcome += f"Downsampling to {width_px} x {height_px}.\n"
        # if start_sec or stop_sec:
        #     # Preparing start-stop filter in cmd
        #     filters.append("setpts=PTS-STARTPTS")
        if filters:
            cmd += ["-vf", ",".join(filters)]

        # CHANGING FPS
        if fps:
            cmd += ["-r", str(fps)]
            outcome += f"Changing fps to {fps}.\n"
        # TRIMMING
        if stop_sec:
            # Setting stop trim filter in cmd
            duration = stop_sec - (start_sec or 0)
            cmd += ["-t", str(duration)]
            outcome += f"Trimming video to {stop_sec} seconds.\n"

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
        # SubprocMixin.run_subproc_fstream(cmd)
        SubprocMixin.run_subproc_console(cmd)
        # Returning outcome
        return outcome

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
