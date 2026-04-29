"""Utility functions."""

import re
import subprocess
from multiprocessing import current_process


def get_cpid() -> int:
    """Get child process ID for multiprocessing."""
    return current_process()._identity[0] if current_process()._identity else 0


def get_gpu_ids():
    """Gets list of GPU IDs from nvidia-smi."""
    try:
        smi_output = subprocess.check_output(
            ["nvidia-smi", "-L"], universal_newlines=True
        )
        gpu_ids = re.findall(r"GPU (\d+):", smi_output)
        gpu_ids = [int(i) for i in gpu_ids]
        return gpu_ids
    except subprocess.CalledProcessError as e:
        # raise e
        print(e)
        return []
