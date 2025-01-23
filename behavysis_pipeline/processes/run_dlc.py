"""
Functions have the following format:

Parameters
----------
vid_fp : str
    The formatted video filepath.
dst_fp : str
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

from behavysis_pipeline.constants import CACHE_DIR
from behavysis_pipeline.df_classes.keypoints_df import KeypointsDf
from behavysis_pipeline.pydantic_models.configs import ExperimentConfigs
from behavysis_pipeline.utils.diagnostics_utils import file_exists_msg
from behavysis_pipeline.utils.io_utils import get_name, silent_remove
from behavysis_pipeline.utils.logging_utils import get_io_obj_content, init_logger_io_obj
from behavysis_pipeline.utils.subproc_utils import run_subproc_console
from behavysis_pipeline.utils.template_utils import save_template

DLC_HDF_KEY = "data"


class RunDLC:
    """_summary_"""

    @classmethod
    def ma_dlc_run_single(
        cls,
        formatted_vid_fp: str,
        keypoints_fp: str,
        configs_fp: str,
        gputouse: int | None,
        overwrite: bool,
    ) -> str:
        """
        Running custom DLC script to generate a DLC keypoints dataframe from a single video.
        """
        logger, io_obj = init_logger_io_obj()
        if not overwrite and os.path.exists(keypoints_fp):
            logger.warning(file_exists_msg(keypoints_fp))
            return get_io_obj_content(io_obj)
        # Getting model_fp
        configs = ExperimentConfigs.read_json(configs_fp)
        model_fp = configs.get_ref(configs.user.run_dlc.model_fp)
        # Derive more parameters
        temp_dlc_dir = os.path.join(CACHE_DIR, f"dlc_{gputouse}")
        keypoints_dir = os.path.dirname(keypoints_fp)
        # Making output directories
        os.makedirs(temp_dlc_dir, exist_ok=True)

        # Assertion: the config.yaml file must exist.
        if not os.path.isfile(model_fp):
            raise ValueError(
                f'The given model_fp file does not exist: "{model_fp}".\n'
                + 'Check this file and specify a DLC ".yaml" config file.'
            )

        # Running the DLC subprocess (in a separate conda env)
        run_dlc_subproc(model_fp, [formatted_vid_fp], temp_dlc_dir, CACHE_DIR, gputouse, logger)

        # Exporting the h5 to chosen file format
        export2df(formatted_vid_fp, temp_dlc_dir, keypoints_dir, logger)
        silent_remove(temp_dlc_dir)

        return get_io_obj_content(io_obj)

    @staticmethod
    def ma_dlc_run_batch(
        vid_fp_ls: list[str],
        keypoints_dir: str,
        configs_dir: str,
        gputouse: int | None,
        overwrite: bool,
    ) -> str:
        """
        Running custom DLC script to generate a DLC keypoints dataframe from a single video.
        """
        logger, io_obj = init_logger_io_obj()

        # Specifying the GPU to use and making the output directory
        # Making output directories
        temp_dlc_dir = os.path.join(CACHE_DIR, f"dlc_{gputouse}")
        os.makedirs(temp_dlc_dir, exist_ok=True)

        # If overwrite is False, filtering for only experiments that need processing
        if not overwrite:
            # Getting only the vid_fp_ls elements that do not exist in keypoints_dir
            vid_fp_ls = [
                vid_fp
                for vid_fp in vid_fp_ls
                if not os.path.exists(os.path.join(keypoints_dir, f"{get_name(vid_fp)}.{KeypointsDf.IO}"))
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
        run_dlc_subproc(model_fp, vid_fp_ls, temp_dlc_dir, CACHE_DIR, gputouse, logger)

        # Exporting the h5 to chosen file format
        for vid_fp in vid_fp_ls:
            export2df(vid_fp, temp_dlc_dir, keypoints_dir, logger)
        silent_remove(temp_dlc_dir)
        return get_io_obj_content(io_obj)


def run_dlc_subproc(
    model_fp: str,
    vid_fp_ls: list[str],
    temp_dlc_dir: str,
    temp_dir: str,
    gputouse: int | None,
    logger: logging.Logger,
) -> None:
    """
    Running the DLC subprocess in a separate process (i.e. separate conda env).

    NOTE: any dlc processing error for each video that occur during the subprocess
    will be logged to the console and the process will continue to the next video.
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
        temp_dlc_dir=temp_dlc_dir,
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


def export2df(name: str, src_dir: str, dst_dir: str, logger: logging.Logger) -> None:
    """
    __summary__
    """
    name = get_name(name)
    # Get the corresponding .h5 filename
    name_fp_ls = [i for i in os.listdir(src_dir) if re.search(rf"^{name}DLC.*\.h5$", i)]
    if len(name_fp_ls) == 0:
        logger.warning(f"No .h5 file found for {name}.")
        return
    elif len(name_fp_ls) == 1:
        name_fp = os.path.join(src_dir, name_fp_ls[0])
        # Reading the .h5 file
        # NOTE: may need DLC_HDF_KEY
        df = pd.DataFrame(pd.read_hdf(name_fp))
        # Imputing na values with 0
        df = df.fillna(0)
        # Writing the file
        KeypointsDf.write(df, os.path.join(dst_dir, f"{name}.{KeypointsDf.IO}"))
        logger.info("Outputted DLC file successfully.")

    else:
        logger.warning(f"Multiple .h5 files found for {name}. Expected only 1.")
