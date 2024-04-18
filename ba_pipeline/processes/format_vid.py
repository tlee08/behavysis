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

import cv2

from ba_package.pipeline.experiment_configs import ConfigsVidMetadata
from ba_package.utils.funcs import (
    read_configs,
    run_subprocess_fstream,
    warning_msg,
    write_configs,
)
from ba_package.pipeline.experiment_configs import ExperimentConfigs


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
            return warning_msg()

        # Finding all necessary config parameters for video formatting
        configs = read_configs(configs_fp, ExperimentConfigs)
        configs_filt = configs.user.format_vid
        width_px = configs_filt.width_px
        height_px = configs_filt.height_px
        fps = configs_filt.fps
        start_sec = configs_filt.start_sec
        stop_sec = configs_filt.stop_sec

        # Constructing ffmpeg command
        cmd = ["ffmpeg", "-i", raw_vid_fp]

        # RESIZING and TRIMMING
        filters = []
        if width_px or height_px:
            filters.append(f"scale={width_px}:{height_px}")
            outcome += f"Downsampling to {width_px} x {height_px}.\n"
        if start_sec or stop_sec:
            filters.append("setpts=PTS-STARTPTS")
        if filters:
            cmd += ["-vf", ",".join(filters)]

        # CHANGING FPS
        if fps:
            cmd += ["-r", str(fps)]
            outcome += f"Changing fps to {fps}.\n"
        if start_sec:
            cmd += ["-ss", str(start_sec)]
            outcome += f"Trimming video from {start_sec} seconds.\n"
        if stop_sec:
            duration = stop_sec - (start_sec if start_sec else 0)
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
            formatted_vid_fp,
        ]
        # Running ffmpeg command
        run_subprocess_fstream(cmd)

        # Saving video metadata to configs dict
        outcome += FormatVid.vid_metadata(
            raw_vid_fp, formatted_vid_fp, configs_fp, overwrite
        )
        return outcome

    @staticmethod
    def vid_metadata(
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
        configs = read_configs(configs_fp, ExperimentConfigs)
        for ftype, fp in (("raw_vid", raw_vid_fp), ("formatted_vid", formatted_vid_fp)):
            try:
                setattr(configs.auto, ftype, FormatVid._vid_metadata(fp))
            except ValueError as e:
                outcome += f"WARNING: {str(e)}\n"
        outcome += "Video metadata stored in config file.\n"
        write_configs(configs, configs_fp)
        return outcome

    @staticmethod
    def _vid_metadata(fp: str) -> ConfigsVidMetadata:
        """
        Finds the video metadata/parameters for either the raw or formatted video.

        Parameters
        ----------
        fp : str
            The video filepath.

        Returns
        -------
        ConfigsVidMetadata
            Object containing video metadata.
        """
        configs_meta = ConfigsVidMetadata()
        cap = cv2.VideoCapture(fp)
        if not cap.isOpened():
            raise ValueError(
                f"The file, {fp}, does not exist or is corrupted. Please check this file."
            )
        configs_meta.height_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        configs_meta.width_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        configs_meta.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        configs_meta.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return configs_meta
