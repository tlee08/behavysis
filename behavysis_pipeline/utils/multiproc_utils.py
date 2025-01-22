"""
Utility functions.
"""

import re
import subprocess
from multiprocessing import current_process

from behavysis_pipeline.utils.logging_utils import init_logger

logger = init_logger(__name__)


def get_cpid() -> int:
    """Get child process ID for multiprocessing."""
    return current_process()._identity[0] if current_process()._identity else 0


def get_gpu_ids():
    """
    gets list of GPU IDs from nvidia-smi
    """
    try:
        smi_output = subprocess.check_output(["nvidia-smi", "-L"], universal_newlines=True)
        gpu_ids = re.findall(r"GPU (\d+):", smi_output)
        gpu_ids = [int(i) for i in gpu_ids]
        return gpu_ids
    except subprocess.CalledProcessError as e:
        logger.error("%s", e)
        return []


def get_best_gpu(gputouse: None | int = None) -> str:
    """
    Picks the best GPU ID from the available GPUs.
    Criteria:
    - If `gputouse` is given, then return that GPU ID
        - If the given `gputouse` is not in the list, then raises an error
    - If `gputouse` is NOT given, then return the first GPU ID
        - If there are no GPUs available, then return None

    Returns the best GPU ID in the form of PyTorch device string.
    """
    # Get list of GPU IDs
    gpu_ids = get_gpu_ids()
    if gputouse is not None:
        # If gputouse is given
        # If the gputouse is not in the list, then raise an error
        assert gputouse in gpu_ids, f"GPU {gputouse} not available in {gpu_ids}"
        # Otherwise, using current `gputouse
        id_ = gputouse
    else:
        # If gputouse is NOT given
        if not gpu_ids:
            # If there are no GPUs available, then using `None`
            id_ = gputouse
        else:
            # Otherwise, return the first GPU ID
            id_ = gpu_ids[0]
    return f"/device:GPU:{id_}" if id_ else "/device:CPU:0"
