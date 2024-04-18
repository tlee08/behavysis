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
from typing import Optional

import pandas as pd

from ba_package.utils.funcs import (
    get_name,
    read_configs,
    warning_msg,
    write_feather,
    get_cpid,
)


#####################################################################
#               DLC ANALYSE VIDEO
#####################################################################


class RunDLD:
    """_summary_"""

    @staticmethod
    def ma_dlc_analyse(
        in_fp: str,
        out_fp: str,
        configs_fp: str,
        gputouse: Optional[int],
        overwrite: bool,
    ) -> str:
        """
        Running custom DLC script to generate a DLC keypoints dataframe from a video.
        """
        cmd = [
            "conda",
            "run",
            "-n",
            "dlc_wrapper_env",
            "python",
            "-m",
            "dlc_subproc_wrapper",
            in_fp,
            out_fp,
            configs_fp,
            # f"{gputouse}",
            f"{overwrite}",
        ]


# class RunDLC:
#     """_summary_"""

#     @staticmethod
#     def ma_dlc_analyse(
#         in_fp: str,
#         out_fp: str,
#         configs_fp: str,
#         gputouse: Optional[int],
#         overwrite: bool,
#     ) -> str:
#         """
#         Run the DLC model on the formatted video to generate a DLC annotated video and DLC file
#         for all experiments.
#         The DLC model's config.yaml filepath must be specified in the
#         `config_path` parameter in the `user` section of the config file.

#         Parameters
#         ----------
#         in_fp : str
#             _description_
#         out_fp : str
#             _description_
#         configs_fp : str
#             _description_
#         gputouse : Optional[int]
#             _description_
#         overwrite : bool
#             _description_

#         Returns
#         -------
#         str
#             _description_

#         Raises
#         ------
#         ValueError
#             The file specified in dlc_config_path does not exist.
#         """
#         # lazy import
#         import deeplabcut

#         outcome = ""

#         # If overwrite is False, checking if we should skip processing
#         if not overwrite and os.path.exists(out_fp):
#             return warning_msg()

#         # Getting necessary config parameters
#         configs = read_configs(configs_fp)

#         dlc_config_path = configs.user.run_dlc.dlc_config_path
#         # Derive more parameters
#         out_dir = os.path.abspath(os.path.split(out_fp)[0])
#         orig_wd = os.getcwd()
#         # Assertion: the config.yaml file must exist.
#         if not os.path.isfile(dlc_config_path):
#             raise ValueError(
#                 f'The given dlc_config_path file does not exist: "{dlc_config_path}".\n'
#                 + 'Check this file and specify a DLC ".yaml" config file.'
#             )
#         # If gputouse is None, then getting best GPU to use
#         if gputouse is None:
#             gputouse = get_best_gpu()
#         # Running DLC pose estimation
#         try:
#             # Making dlc file
#             deeplabcut.analyze_videos(
#                 config=dlc_config_path,
#                 videos=in_fp,
#                 videotype=".mp4",
#                 destfolder=out_dir,
#                 auto_track=True,
#                 gputouse=gputouse,
#                 save_as_csv=False,
#                 calibrate=False,
#                 identity_only=False,
#                 allow_growth=False,  # TODO: should this be True?
#             )
#             # # Using the DLC filter predictions process
#             # deeplabcut.filterpredictions(
#             #     dlc_config_path,
#             #     video_path,
#             #     destfolder=destfolder,
#             #     filtertype="median",
#             #     save_as_csv=False,
#             # )
#         except Exception as e:
#             raise ValueError(
#                 f"ERROR: DLC model failed to run due to the following error: {e}"
#             ) from e
#         finally:
#             os.chdir(orig_wd)
#             # Renaming files corresponding to the experiment
#             outcome += clean_raw_dlc_files(out_fp)
#         return outcome


# def get_best_gpu() -> Optional[int]:
#     """
#     Returns the GPU ID of the best GPU to use.
#     The "best" is chosen as:
#     * A GPU ID that exists.
#     * If using multiprocessing, the GPU ID that corresponds to the relative child PID.
#     * Otherwise (i.e. single-processing), the first GPU ID.

#     Returns
#     -------
#     Optional[int]
#         The GPU ID.
#     """
#     # lazy import
#     from tensorflow.config import list_physical_devices

#     # Getting list of available GPUs
#     tf_gpus = list_physical_devices("GPU")
#     tf_gpu_ids = [int(i.name.split(":")[-1]) for i in tf_gpus]
#     # Getting GPU with least usage (if there is one)
#     if len(tf_gpu_ids) == 0:  # If no GPUs, then return None
#         return None
#     # Getting current multiprocess child pid (i.e. the gpu ID to use)
#     gputouse = get_cpid()
#     print("GPU:", gputouse)
#     # If gputouse does not have corresponding GPU ID
#     if gputouse not in tf_gpu_ids:
#         gputouse = 0
#     return gputouse


# def clean_raw_dlc_files(out_fp: str) -> str:
#     """
#     Cleaning up the DLC files for the given filepath.
#     This involves:
#     - Converting the corresponding outputted .h5 dataframe to a .feather file.
#     - Removing all other corresponding files in the `out_fp` directory.

#     Parameters
#     ----------
#     out_fp : str
#         _description_

#     Returns
#     -------
#     str
#         _description_
#     """
#     outcome = ""
#     # Renaming files corresponding to the experiment
#     destfolder = os.path.abspath(os.path.split(out_fp)[0])
#     name = get_name(out_fp)
#     # Iterating through all files in the outpur directory
#     for fp in os.listdir(destfolder):
#         # Looking at only files corresponding to the experiment (by name)
#         if re.search(rf"^{name}DLC", fp):
#             if re.search(r"\.h5$", fp):
#                 # copying file to dlc folder
#                 df = pd.DataFrame(pd.read_hdf(os.path.join(destfolder, fp)))
#                 write_feather(df, os.path.join(destfolder, f"{name}.feather"))
#             # Deleting original DLC file
#             os.remove(os.path.join(destfolder, fp))
#     outcome += (
#         "DLC output files have been renamed and placed in corresponding folders.\n"
#     )
#     return outcome
