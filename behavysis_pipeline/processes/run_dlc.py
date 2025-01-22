"""
Functions have the following format:

Parameters
----------
vid_fp : str
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

from behavysis_pipeline.constants import CACHE_DIR
from behavysis_pipeline.df_classes.keypoints_df import KeypointsDf
from behavysis_pipeline.pydantic_models.configs import ExperimentConfigs
from behavysis_pipeline.utils.diagnostics_utils import file_exists_msg
from behavysis_pipeline.utils.io_utils import get_name, silent_remove
from behavysis_pipeline.utils.logging_utils import get_io_obj_content, init_logger_with_io_obj
from behavysis_pipeline.utils.misc_utils import enum2tuple, get_current_func_name
from behavysis_pipeline.utils.subproc_utils import run_subproc_console
from behavysis_pipeline.utils.template_utils import save_template

DLC_HDF_KEY = "data"


class RunDLC:
    """_summary_"""

    @classmethod
    def ma_dlc_analyse_single(
        cls,
        vid_fp: str,
        out_fp: str,
        configs_fp: str,
        gputouse: int | None,
        overwrite: bool,
    ) -> str:
        """
        Running custom DLC script to generate a DLC keypoints dataframe from a single video.
        """
        logger, io_obj = init_logger_with_io_obj(get_current_func_name())
        if not overwrite and os.path.exists(out_fp):
            logger.warning(file_exists_msg(out_fp))
            return get_io_obj_content(io_obj)
        # Getting model_fp
        configs = ExperimentConfigs.read_json(configs_fp)
        model_fp = configs.get_ref(configs.user.run_dlc.model_fp)
        # Derive more parameters
        dlc_out_dir = os.path.join(CACHE_DIR, f"dlc_{gputouse}")
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
        run_dlc_subproc(model_fp, [vid_fp], dlc_out_dir, CACHE_DIR, gputouse)

        # Exporting the h5 to chosen file format in the out_dir
        logger.info(export2df(vid_fp, dlc_out_dir, out_dir))
        # silent_remove(dlc_out_dir)

        return get_io_obj_content(io_obj)

    @staticmethod
    def ma_dlc_analyse_batch(
        vid_fp_ls: list[str],
        out_dir: str,
        configs_dir: str,
        gputouse: int | None,
        overwrite: bool,
    ) -> str:
        """
        Running custom DLC script to generate a DLC keypoints dataframe from a single video.
        """
        logger, io_obj = init_logger_with_io_obj(get_current_func_name())

        # Specifying the GPU to use and making the output directory
        # Making output directories
        dlc_out_dir = os.path.join(CACHE_DIR, f"dlc_{gputouse}")
        os.makedirs(dlc_out_dir, exist_ok=True)

        # If overwrite is False, filtering for only experiments that need processing
        if not overwrite:
            # Getting only the vid_fp_ls elements that do not exist in out_dir
            vid_fp_ls = [
                vid_fp
                for vid_fp in vid_fp_ls
                if not os.path.exists(os.path.join(out_dir, f"{get_name(vid_fp)}.{KeypointsDf.IO}"))
            ]

        # If there are no videos to process, return
        if len(vid_fp_ls) == 0:
            return get_io_obj_content(io_obj)

        # Getting the DLC model config path
        # Getting the names of the files that need processing
        dlc_fp_ls = [get_name(i) for i in vid_fp_ls]
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
        run_dlc_subproc(model_fp, vid_fp_ls, dlc_out_dir, CACHE_DIR, gputouse)

        # Exporting the h5 to chosen file format in the out_dir
        for vid_fp in vid_fp_ls:
            logger.info(export2df(vid_fp, dlc_out_dir, out_dir))
        silent_remove(dlc_out_dir)
        return get_io_obj_content(io_obj)


def run_dlc_subproc(
    model_fp: str,
    vid_fp_ls: list[str],
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
    save_template(
        "dlc_subproc.py",
        "behavysis_pipeline",
        "templates",
        script_fp,
        vid_fp_ls=vid_fp_ls,
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
    # run_subproc_fstream(cmd)
    run_subproc_console(cmd)
    # Removing the script file
    silent_remove(script_fp)


def export2df(name: str, in_dir: str, out_dir: str) -> str:
    """
    __summary__
    """
    # Get name
    name = get_name(name)
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
        df.index.names = list(enum2tuple(KeypointsDf.IN))
        df.columns.names = list(enum2tuple(KeypointsDf.CN))
        # Imputing na values with 0
        df = df.fillna(0)
        # Writing the file
        KeypointsDf.write(df, os.path.join(out_dir, f"{name}.{KeypointsDf.IO}"))
        return "Outputted DLC file successfully."
    else:
        raise ValueError(f"Multiple .h5 files found for {name}.")
