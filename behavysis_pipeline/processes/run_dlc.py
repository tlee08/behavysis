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

import pandas as pd
from behavysis_core.df_classes.df_mixin import DFMixin
from behavysis_core.df_classes.keypoints_df import KeypointsDf
from behavysis_core.mixins.io_mixin import IOMixin
from behavysis_core.mixins.subproc_mixin import SubprocMixin
from behavysis_core.pydantic_models.experiment_configs import ExperimentConfigs

DLC_HDF_KEY = "data"


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
        # Getting model_fp
        configs = ExperimentConfigs.read_json(configs_fp)
        model_fp = configs.get_ref(configs.user.run_dlc.model_fp)
        # Derive more parameters
        dlc_out_dir = os.path.join(temp_dir, f"dlc_{gputouse}")
        out_dir = os.path.dirname(out_fp)
        # Making output directories
        os.makedirs(dlc_out_dir, exist_ok=True)

        # Assertion: the config.yaml file must exist.
        if not os.path.isfile(model_fp):
            raise ValueError(
                f'The given model_fp file does not exist: "{model_fp}".\n'
                + 'Check this file and specify a DLC ".yaml" config file.'
            )

        # Running the DLC subprocess (in a separate conda env)
        run_dlc_subproc(model_fp, [in_fp], dlc_out_dir, temp_dir, gputouse)

        # Exporting the h5 to feather the out_dir
        export_2_feather(in_fp, dlc_out_dir, out_dir)
        # IOMixin.silent_rm(dlc_out_dir)

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

        # Specifying the GPU to use and making the output directory
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

        # If there are no videos to process, return
        if len(in_fp_ls) == 0:
            return outcome

        # Getting the DLC model config path
        # Getting the names of the files that need processing
        dlc_fp_ls = [IOMixin.get_name(i) for i in in_fp_ls]
        # Getting their corresponding configs_fp
        dlc_fp_ls = [os.path.join(configs_dir, f"{i}.json") for i in dlc_fp_ls]
        # Reading their configs
        dlc_fp_ls = [ExperimentConfigs.read_json(i) for i in dlc_fp_ls]
        # Getting their model_fp
        dlc_fp_ls = [i.user.run_dlc.model_fp for i in dlc_fp_ls]
        # Converting to a set
        dlc_fp_set = set(dlc_fp_ls)
        # Assertion: all model_fp must be the same
        assert len(dlc_fp_set) == 1
        # Getting the model_fp
        model_fp = dlc_fp_set.pop()
        # Assertion: the config.yaml file must exist.
        assert os.path.isfile(model_fp), (
            f'The given model_fp file does not exist: "{model_fp}".\n'
            + 'Check this file and specify a DLC ".yaml" config file.'
        )

        # Running the DLC subprocess (in a separate conda env)
        run_dlc_subproc(model_fp, in_fp_ls, dlc_out_dir, temp_dir, gputouse)

        # Exporting the h5 to feather the out_dir
        for in_fp in in_fp_ls:
            outcome += export_2_feather(in_fp, dlc_out_dir, out_dir)
        IOMixin.silent_rm(dlc_out_dir)
        # Returning outcome
        return outcome


def run_dlc_subproc(
    model_fp: str,
    in_fp_ls: list[str],
    dlc_out_dir: str,
    temp_dir: str,
    gputouse: int | None,
):
    """
    Running the DLC subprocess in a separate process (i.e. separate conda env).

    NOTE: any dlc processing error for each video that occur during the subprocess
    will be printed to the console and the process will continue to the next video.
    """
    # TODO: implement for and try for each video and get errors?? Maybe save a log to a file
    # Saving the script to a file
    script_fp = os.path.join(temp_dir, f"dlc_subproc_{gputouse}.py")
    IOMixin.save_template(
        "dlc_subproc.py",
        "behavysis_pipeline",
        "templates",
        script_fp,
        in_fp_ls=in_fp_ls,
        model_fp=model_fp,
        dlc_out_dir=dlc_out_dir,
        gputouse=gputouse,
    )
    # Running the DLC subprocess in a separate conda env
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


def export_2_feather(name: str, in_dir: str, out_dir: str) -> str:
    """
    __summary__
    """
    # Get name
    name = IOMixin.get_name(name)
    # Get the corresponding .h5 filename
    name_fp_ls = [i for i in os.listdir(in_dir) if re.search(rf"^{name}DLC.*\.h5$", i)]
    if len(name_fp_ls) == 0:
        return f"WARNING: No .h5 file found for {name}."
    elif len(name_fp_ls) == 1:
        name_fp = os.path.join(in_dir, name_fp_ls[0])
        # Reading the .h5 file
        # NOTE: may need DLC_HDF_KEY
        df = pd.DataFrame(pd.read_hdf(name_fp))
        # Setting the column and index level names
        df.index.names = list(DFMixin.enum2tuple(KeypointsDf.IN))
        df.columns.names = list(DFMixin.enum2tuple(KeypointsDf.CN))
        # Imputing na values with 0
        df = df.fillna(0)
        # Checking df
        KeypointsDf.check_df(df)
        # Writing the .feather file
        DFMixin.write_feather(df, os.path.join(out_dir, f"{name}.feather"))
        return "Outputted DLC file successfully."
    else:
        raise ValueError(f"Multiple .h5 files found for {name}.")
