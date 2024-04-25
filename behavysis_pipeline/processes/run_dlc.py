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
import re
import shutil
from typing import Optional

import numpy as np
import pandas as pd
from behavysis_core.mixins.df_io_mixin import DFIOMixin
from behavysis_core.mixins.diagnostics_mixin import DiagnosticsMixin
from behavysis_core.mixins.io_mixin import IOMixin
from behavysis_core.mixins.multiproc_mixin import MultiprocMixin
from behavysis_core.mixins.subproc_mixin import SubprocMixin

from behavysis_pipeline.pipeline.experiment_configs import ExperimentConfigs

# from tensorflow.config import list_physical_devices


#####################################################################
#               DLC ANALYSE VIDEO
#####################################################################


class RunDLC:
    """_summary_"""

    @staticmethod
    def ma_dlc_analyse_single(
        in_fp: str,
        out_fp: str,
        configs_fp: str,
        temp_dir: str,
        gputouse: int | None,
        overwrite: bool,
    ) -> str:
        """
        Running custom DLC script to generate a DLC keypoints dataframe from a single video.
        """
        outcome = ""

        # If overwrite is False, checking if we should skip processing
        if not overwrite and os.path.exists(out_fp):
            return DiagnosticsMixin.warning_msg()

        # Getting dlc_config_path
        configs = ExperimentConfigs.read_json(configs_fp)
        dlc_config_path = configs.user.run_dlc.dlc_config_path
        # Derive more parameters
        dlc_out_dir = os.path.join(temp_dir, f"dlc_{gputouse}")
        out_dir = os.path.dirname(out_fp)
        # Making output directories
        os.makedirs(dlc_out_dir, exist_ok=True)

        # Assertion: the config.yaml file must exist.
        if not os.path.isfile(dlc_config_path):
            raise ValueError(
                f'The given dlc_config_path file does not exist: "{dlc_config_path}".\n'
                + 'Check this file and specify a DLC ".yaml" config file.'
            )

        subproc_run_dlc(dlc_config_path, [in_fp], dlc_out_dir, temp_dir, gputouse)

        # Cleaning up the DLC files
        clean_raw_dlc_files(in_fp, dlc_out_dir, out_dir)

        return outcome

    @staticmethod
    def ma_dlc_analyse_batch(
        in_fp_ls: list[str],
        out_dir: str,
        configs_dir: str,
        temp_dir: str,
        gputouse: int | None,
        overwrite: bool,
    ) -> str:
        """
        Running custom DLC script to generate a DLC keypoints dataframe from a single video.
        """
        outcome = ""

        # Getting the child process ID (i.e. corresponds to the GPU ID to use)
        if gputouse is None:
            gpu_ids = MultiprocMixin.get_gpu_ids()
            gputouse = np.max([MultiprocMixin.get_cpid() - 1, 0]) % len(gpu_ids)

        # Assertion: the dlc_config_path should be the same for all configs_fp_ls
        # TODO

        # Getting dlc_config_path
        configs_fp_0 = os.path.join(
            configs_dir, f"{IOMixin.get_name(in_fp_ls[0])}.json"
        )
        configs = ExperimentConfigs.read_json(configs_fp_0)
        dlc_config_path = configs.user.run_dlc.dlc_config_path
        # Derive more parameters
        dlc_out_dir = os.path.join(temp_dir, f"dlc_{gputouse}")
        # Making output directories
        os.makedirs(dlc_out_dir, exist_ok=True)

        # Assertion: the config.yaml file must exist.
        if not os.path.isfile(dlc_config_path):
            raise ValueError(
                f'The given dlc_config_path file does not exist: "{dlc_config_path}".\n'
                + 'Check this file and specify a DLC ".yaml" config file.'
            )

        subproc_run_dlc(dlc_config_path, in_fp_ls, dlc_out_dir, temp_dir, gputouse)

        # Cleaning up the DLC files
        for in_fp in in_fp_ls:
            clean_raw_dlc_files(in_fp, dlc_out_dir, out_dir)

        return outcome


def subproc_run_dlc(
    dlc_config_path: str,
    in_fp_ls: list[str],
    dlc_out_dir: str,
    temp_dir: str,
    gputouse: int,
):
    """Running the DLC subprocess in a separate process (i.e. separate conda env)."""
    # Generating a script to run the DLC analysis
    script_fp = os.path.join(temp_dir, f"script_{gputouse}.py")
    with open(script_fp, "w", encoding="utf-8") as f:
        f.write(
            f"""
import deeplabcut

deeplabcut.analyze_videos(
    config=r'{dlc_config_path}',
    videos={in_fp_ls},
    videotype='mp4',
    destfolder=r'{dlc_out_dir}',
    gputouse={gputouse},
    save_as_csv=False,
    calibrate=False,
    identity_only=False,
    allow_growth=False,
)
"""
        )

    cmd = [
        os.environ["CONDA_EXE"],
        "run",
        "--no-capture-output",
        "-n",
        "DEEPLABCUT",
        "python",
        script_fp,
    ]
    print(" ".join(cmd))
    # SubprocMixin.run_subproc_fstream(cmd)
    SubprocMixin.run_subproc_console(cmd)


def clean_raw_dlc_files(in_fp, dlc_out_dir, out_dir: str) -> str:
    """
    Cleaning up the DLC files for the given filepath.
    This involves:
    - Converting the corresponding outputted .h5 dataframe to a .feather file.
    - Removing all other corresponding files in the `out_fp` directory.

    Parameters
    ----------
    out_fp : str
        _description_

    Returns
    -------
    str
        _description_
    """
    outcome = ""
    # Renaming files corresponding to the experiment
    name = IOMixin.get_name(in_fp)

    # Iterating through all files in the dlc_out_dir directory
    for fp in os.listdir(dlc_out_dir):
        # Looking at only files corresponding to the experiment (by name)
        if re.search(rf"^{name}DLC", fp):
            if re.search(r"\.h5$", fp):
                # copying file to dlc folder
                df = pd.DataFrame(pd.read_hdf(os.path.join(dlc_out_dir, fp)))
                DFIOMixin.write_feather(df, os.path.join(out_dir, f"{name}.feather"))
            # Deleting original DLC file
            os.remove(os.path.join(dlc_out_dir, fp))
    outcome += (
        "DLC output files have been renamed and placed in corresponding folders.\n"
    )
    return outcome
    return outcome
