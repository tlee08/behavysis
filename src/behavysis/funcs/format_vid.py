"""Video formatting with ffmpeg."""

import subprocess
from pathlib import Path

import cv2
from loguru import logger

from behavysis.models import ExperimentConfigs
from behavysis.models.funcs import VidMetadata
from behavysis.utils.io_utils import file_exists_msg


def format_vid(
    raw_vid_fp: Path,
    formatted_vid_fp: Path,
    configs_fp: Path,
    *,
    overwrite: bool,
) -> None:
    """Format video with ffmpeg and save metadata to configs."""
    if not overwrite and formatted_vid_fp.exists():
        logger.warning(file_exists_msg(formatted_vid_fp))
        return

    configs = ExperimentConfigs.model_validate_json(configs_fp.read_text())
    cfg = configs.user.format_vid

    # Build ffmpeg command
    cmd = ["ffmpeg"]

    if cfg.start_sec:
        cmd += ["-ss", str(configs.get_ref(cfg.start_sec))]

    cmd += ["-i", str(raw_vid_fp)]

    filters = []
    width = configs.get_ref(cfg.width_px)
    height = configs.get_ref(cfg.height_px)
    if width or height:
        filters.append(f"scale={width or -1}:{height or -1}")
    if filters:
        cmd += ["-vf", ",".join(filters)]

    if cfg.fps:
        cmd += ["-r", str(configs.get_ref(cfg.fps))]

    if cfg.stop_sec:
        duration = configs.get_ref(cfg.stop_sec) - (configs.get_ref(cfg.start_sec) or 0)
        cmd += ["-t", str(duration)]

    cmd += [
        "-c:v",
        "h264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-y",
        str(formatted_vid_fp),
    ]

    formatted_vid_fp.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)

    # Save metadata to configs
    configs.auto.raw_vid = _get_vid_metadata(raw_vid_fp)
    configs.auto.formatted_vid = _get_vid_metadata(formatted_vid_fp)
    configs_fp.write_text(configs.model_dump_json(indent=2))


def _get_vid_metadata(vid_fp: Path) -> VidMetadata:
    """Extract metadata from video file."""
    meta = VidMetadata()
    cap = cv2.VideoCapture(vid_fp)
    if cap.isOpened():
        meta.height_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        meta.width_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        meta.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        meta.fps = cap.get(cv2.CAP_PROP_FPS)
    else:
        logger.warning("Cannot open video: %s", vid_fp)
    cap.release()
    return meta
