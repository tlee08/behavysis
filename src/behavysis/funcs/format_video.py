"""Video formatting with ffmpeg."""

import subprocess
from pathlib import Path

import cv2
from loguru import logger

from behavysis.models import ExperimentConfig, ExperimentMetadata, VideoMetadata


def format_video(
    raw_vid_fp: Path,
    formatted_vid_fp: Path,
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
) -> ExperimentMetadata:
    """Format video with ffmpeg and save metadata to config."""
    cfg = config.require_format_video()
    # Build ffmpeg command
    cmd = ["ffmpeg"]
    if cfg.start_sec:
        cmd += ["-ss", str(cfg.start_sec)]
    cmd += ["-i", str(raw_vid_fp)]
    filters = []
    width = cfg.width_px
    height = cfg.height_px
    if width or height:
        filters.append(f"scale={width or -1}:{height or -1}")
    if filters:
        cmd += ["-vf", ",".join(filters)]
    if cfg.fps:
        cmd += ["-r", str(cfg.fps)]
    if cfg.stop_sec:
        duration = cfg.stop_sec - (cfg.start_sec or 0)
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
    # Running format vid with ffmpeg
    subprocess.run(cmd, check=True)
    # Save metadata to config
    # Always do this
    metadata.raw_video = get_vid_metadata(raw_vid_fp)
    metadata.formatted_video = get_vid_metadata(formatted_vid_fp)
    return metadata


def get_vid_metadata(vid_fp: Path) -> VideoMetadata:
    """Extract metadata from video file."""
    meta = VideoMetadata()
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
