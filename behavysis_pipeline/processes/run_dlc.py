"""
Functions have the following format:

Parameters
----------
in_fp : str
    The formatted video filepath.
out_fp : str
    The dlc output filepath.
configs_fp : str
    The JSON configs filepath.
gputouse : int
    The GPU's number so computation is done on this GPU.
    If None, then tries to select the best GPU (if it exists).
overwrite : bool
    Whether to overwrite the output file (if it exists).

Returns
-------
str
    The outcome of the process.
"""

import os

import numpy as np
from behavysis_core.mixins.subprocess_mixin import SubprocessMixin

from behavysis_pipeline.pipeline.experiment_configs import ExperimentConfigs

#####################################################################
#               DLC ANALYSE VIDEO
#####################################################################


class RunDLC:
    """_summary_"""

    @staticmethod
    def ma_dlc_analyse(
        in_fp: str,
        out_fp: str,
        configs_fp: str,
        gputouse: int | None,
        overwrite: bool,
    ) -> str:
        """
        Running custom DLC script to generate a DLC keypoints dataframe from a video.
        """
        outcome = ""
        # Getting the child process ID (i.e. corresponds to the GPU ID to use)
        if gputouse is None:
            gputouse = np.max([SubprocessMixin.get_cpid() - 1, 0])
        # Getting dlc_config_path
        configs = ExperimentConfigs.read_json(configs_fp)
        dlc_config_path = configs.user.run_dlc.dlc_config_path
        # Running the DLC subprocess
        cmd = [
            os.environ["CONDA_EXE"],
            "run",
            "--no-capture-output",
            "-n",
            "dlc_subproc_env",
            "dlc_subproc",
            in_fp,
            out_fp,
            dlc_config_path,
            f"{gputouse}",
            f"{overwrite}",
        ]
        # SubprocessMixin.run_subprocess_fstream(cmd)
        SubprocessMixin.run_subprocess_console(cmd)
        return outcome
        return outcome
