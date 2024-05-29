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

import logging
import os
import re

import pandas as pd
from behavysis_core.constants import DLC_COLUMN_NAMES, DLC_INDEX_NAME
from behavysis_core.data_models.experiment_configs import ExperimentConfigs
from behavysis_core.mixins.df_io_mixin import DFIOMixin
from behavysis_core.mixins.io_mixin import IOMixin
from behavysis_core.mixins.subproc_mixin import SubprocMixin


class RunDLC:
    """_summary_"""

    @staticmethod
    @IOMixin.overwrite_check()
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
        # Specifying the GPU to use
        gputouse = "None" if not gputouse else gputouse
        # Getting dlc_config_path
        configs = ExperimentConfigs.read_json(configs_fp)
        dlc_config_path = configs.get_ref(configs.user.run_dlc.dlc_config_path)
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

        # Running the DLC subprocess (in a separate conda env)
        subproc_run_dlc(dlc_config_path, [in_fp], dlc_out_dir, temp_dir, gputouse)

        # Exporting the h5 to feather the out_dir
        export_dlc_to_feather(in_fp, dlc_out_dir, out_dir)
        IOMixin.silent_rm(dlc_out_dir)

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

        # Specifying the GPU to use
        # and making the output directory
        if not gputouse:
            gputouse = "None"
        # Making output directories
        dlc_out_dir = os.path.join(temp_dir, f"dlc_{gputouse}")
        os.makedirs(dlc_out_dir, exist_ok=True)

        # If overwrite is False, filtering for only experiments that need processing
        if not overwrite:
            # Getting only the in_fp_ls elements that do not exist in out_dir
            in_fp_ls = [
                i
                for i in in_fp_ls
                if not os.path.exists(
                    os.path.join(out_dir, f"{IOMixin.get_name(i)}.feather")
                )
            ]

        # Getting the DLC model config path
        # Getting the names of the files that need processing
        dlc_fp_ls = [IOMixin.get_name(i) for i in in_fp_ls]
        # Getting their corresponding configs_fp
        dlc_fp_ls = [os.path.join(configs_dir, f"{i}.json") for i in dlc_fp_ls]
        # Reading their configs
        dlc_fp_ls = [ExperimentConfigs.read_json(i) for i in dlc_fp_ls]
        # Getting their dlc_config_path
        dlc_fp_ls = [i.user.run_dlc.dlc_config_path for i in dlc_fp_ls]
        # Converting to a set
        dlc_fp_set = set(dlc_fp_ls)
        # Assertion: all dlc_config_paths must be the same
        assert len(dlc_fp_set) == 1
        # Getting the dlc_config_path
        dlc_config_path = dlc_fp_set.pop()
        # Assertion: the config.yaml file must exist.
        assert os.path.isfile(dlc_config_path), (
            f'The given dlc_config_path file does not exist: "{dlc_config_path}".\n'
            + 'Check this file and specify a DLC ".yaml" config file.'
        )

        # Running the DLC subprocess (in a separate conda env)
        subproc_run_dlc(dlc_config_path, in_fp_ls, dlc_out_dir, temp_dir, gputouse)

        # Exporting the h5 to feather the out_dir
        for in_fp in in_fp_ls:
            export_dlc_to_feather(in_fp, dlc_out_dir, out_dir)
        IOMixin.silent_rm(dlc_out_dir)
        # Returning outcome
        return outcome


def subproc_run_dlc(
    dlc_config_path: str,
    in_fp_ls: list[str],
    dlc_out_dir: str,
    temp_dir: str,
    gputouse: int,
):
    """
    Running the DLC subprocess in a separate process (i.e. separate conda env).

    NOTE: any dlc processing error for each video that occur during the subprocess
    will be printed to the console and the process will continue to the next video.
    """
    # Generating a script to run the DLC analysis
    # TODO: implement for and try for each video and get errors?? Maybe save a log to a file
    script_fp = os.path.join(temp_dir, f"script_{gputouse}.py")
    with open(script_fp, "w", encoding="utf-8") as f:
        f.write(
            f"""
import deeplabcut

for video in {in_fp_ls}:
    try:
        deeplabcut.analyze_videos(
            config=r'{dlc_config_path}',
            videos=[video],
            videotype='mp4',
            destfolder=r'{dlc_out_dir}',
            gputouse={gputouse},
            save_as_csv=False,
            calibrate=False,
            identity_only=False,
            allow_growth=False,
        )
    except Exception as e:
        print(f'Error', e)
"""
        )

    # Running the DLC subprocess
    cmd = [
        os.environ["CONDA_EXE"],
        "run",
        "--no-capture-output",
        "-n",
        "DEEPLABCUT",
        "python",
        script_fp,
    ]
    # SubprocMixin.run_subproc_fstream(cmd)
    SubprocMixin.run_subproc_console(cmd)
    # Removing the script file
    IOMixin.silent_rm(script_fp)


def export_dlc_to_feather(name: str, in_dir: str, out_dir: str) -> str:
    """
    __summary__
    """
    # Get name
    name = IOMixin.get_name(name)
    # Get the corresponding .h5 filename
    name_fp_ls = [i for i in os.listdir(in_dir) if re.search(rf"^{name}DLC.*\.h5$", i)]
    if len(name_fp_ls) == 0:
        logging.info("No .h5 file found for %s.", name)
    elif len(name_fp_ls) == 1:
        name_fp = os.path.join(in_dir, name_fp_ls[0])
        # Reading the .h5 file
        df = pd.DataFrame(pd.read_hdf(name_fp))
        # Setting the column and index level names
        df.index.name = DLC_INDEX_NAME
        df.columns.names = DLC_COLUMN_NAMES
        # Imputing na values with 0
        df = df.fillna(0)
        # Writing the .feather file
        DFIOMixin.write_feather(df, os.path.join(out_dir, f"{name}.feather"))
    else:
        raise ValueError(f"Multiple .h5 files found for {name}.")
