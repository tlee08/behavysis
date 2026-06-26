"""Utility functions."""

import re
import subprocess

from loguru import logger


def get_gpu_ids() -> list[int]:
    """Gets list of GPU IDs from nvidia-smi."""
    try:
        smi_output = subprocess.check_output(
            ["nvidia-smi", "-L"],
            universal_newlines=True,
        )
        gpu_ids = re.findall(r"GPU (\d+):", smi_output)
        return [int(i) for i in gpu_ids]
    except subprocess.CalledProcessError as e:
        logger.exception(e)
        return []
