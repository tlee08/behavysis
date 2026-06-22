"""Functions have the following format."""

import os
import re
import subprocess
from pathlib import Path

import pandas as pd
from loguru import logger

from behavysis.constants import CACHE_DIR
from behavysis.df_classes.keypoints_df import CoordsCols, KeypointsDf
from behavysis.models.experiment_configs import ExperimentConfigs
from behavysis.utils.io_utils import file_exists_msg, silent_remove
from behavysis.utils.template_utils import save_template

DLC_HDF_KEY = "data"


def ma_dlc_run_single(
    formatted_vid_fp: Path,
    keypoints_fp: Path,
    configs_fp: Path,
    gputouse: int | None,
    *,
    overwrite: bool,
) -> None:
    """Running DLC script to generate a keypoints dataframe from a single video."""
    if not overwrite and keypoints_fp.exists():
        logger.warning(file_exists_msg(keypoints_fp))
        return
    # Getting model_fp
    configs = ExperimentConfigs.model_validate_json(configs_fp.read_text())
    model_fp = configs.get_ref(configs.user.run_dlc.model_fp)
    # Derive more parameters
    temp_dlc_dir = CACHE_DIR / f"dlc_{gputouse}"
    keypoints_dir = keypoints_fp.parent
    # Making output directories
    temp_dlc_dir.mkdir(parents=True, exist_ok=True)

    # Assertion: the config.yaml file must exist.
    if not model_fp.is_file():
        msg = (
            f'DLC model config not found: "{model_fp}"\n'
            f"  Check user.run_dlc.model_fp in your config file.\n"
            f"  It should point to a DeepLabCut config.yaml file."
        )
        raise ValueError(msg)

    # Running the DLC subprocess (in a separate conda env)
    _run_dlc_subproc(model_fp, [formatted_vid_fp], temp_dlc_dir, CACHE_DIR, gputouse)

    # Exporting the h5 to chosen file format
    _export2df(formatted_vid_fp.stem, temp_dlc_dir, keypoints_dir)
    silent_remove(temp_dlc_dir)


def ma_dlc_run_batch(
    vid_fp_ls: list[Path],
    keypoints_dir: Path,
    configs_dir: Path,
    gputouse: int | None,
    *,
    overwrite: bool,
) -> None:
    """Running DLC to generate a keypoints dataframe from a single video."""
    # Specifying the GPU to use and making the output directory
    # Making output directories
    temp_dlc_dir = CACHE_DIR / f"dlc_{gputouse}"
    temp_dlc_dir.mkdir(parents=True, exist_ok=True)

    # If overwrite is False, filtering for only experiments that need processing
    if not overwrite:
        # Getting only the vid_fp_ls elements that do not exist in keypoints_dir
        vid_fp_ls = [
            vid_fp
            for vid_fp in vid_fp_ls
            if not (keypoints_dir / f"{vid_fp.stem}.{KeypointsDf.IO}").exists()
        ]

    # If there are no videos to process, return
    if len(vid_fp_ls) == 0:
        return

    # Getting the DLC model config path
    # Getting the names of the files that need processing
    dlc_fp_ls = [i.stem for i in vid_fp_ls]
    # Getting their corresponding configs_fp
    dlc_fp_ls = [configs_dir / f"{i}.json" for i in dlc_fp_ls]
    # Reading their configs
    dlc_fp_ls = [
        ExperimentConfigs.model_validate_json(i.read_text()) for i in dlc_fp_ls
    ]
    # Getting their model_fp
    dlc_fp_ls = [i.user.run_dlc.model_fp for i in dlc_fp_ls]
    # Converting to a set
    dlc_fp_set = set(dlc_fp_ls)
    # Assertion: all model_fp must be the same
    assert len(dlc_fp_set) == 1
    # Getting the model_fp
    model_fp = dlc_fp_set.pop()
    # Assertion: the config.yaml file must exist.
    assert model_fp.is_file(), (
        f'DLC model config not found: "{model_fp}"\n'
        f"  Check user.run_dlc.model_fp in your config files.\n"
        f"  All experiments in this batch must use the same model."
    )

    # Running the DLC subprocess (in a separate conda env)
    _run_dlc_subproc(model_fp, vid_fp_ls, temp_dlc_dir, CACHE_DIR, gputouse)

    # Exporting the h5 to chosen file format
    for vid_fp in vid_fp_ls:
        _export2df(vid_fp.stem, temp_dlc_dir, keypoints_dir)
    silent_remove(temp_dlc_dir)


def _run_dlc_subproc(
    model_fp: Path,
    vid_fp_ls: list[Path],
    temp_dlc_dir: Path,
    temp_dir: Path,
    gputouse: int | None,
) -> None:
    """Running the DLC subprocess in a separate process (i.e. separate conda env).

    NOTE: any dlc processing error for each video that occur during the subprocess
    will be logged to the console and the process will continue to the next video.
    """
    # Saving the script to a file.
    script_fp = temp_dir / f"dlc_subproc_{gputouse}.py"
    save_template(
        "dlc_subproc.py",
        CACHE_DIR / "dlc_subproc.py",
        vid_fp_ls=vid_fp_ls,
        model_fp=model_fp,
        temp_dlc_dir=temp_dlc_dir,
        gputouse=gputouse,
    )
    logger.info("Running the DLC subprocess in a separate conda environment.")
    cmd = [
        os.environ["CONDA_EXE"],
        "run",
        "--no-capture-output",
        "-n",
        "DEEPLABCUT",
        "python",
        str(script_fp),
    ]
    subprocess.run(cmd, check=True)
    silent_remove(script_fp)


def _export2df(name: str, src_dir: Path, dst_dir: Path) -> None:
    """Export DLC h5 output to project dataframe format."""
    # Get the corresponding .h5 filename
    name_fp_ls = [
        i for i in src_dir.iterdir() if re.search(rf"^{name}DLC.*\.h5$", i.name)
    ]
    if len(name_fp_ls) == 0:
        msg = f"No .h5 file found for {name}."
        logger.warning(msg)
        return
    if len(name_fp_ls) == 1:
        name_fp = src_dir / name_fp_ls[0]
        # Reading the .h5 file
        # NOTE: may need DLC_HDF_KEY
        df = pd.DataFrame(pd.read_hdf(name_fp))
        # Imputing na values with 0
        df = df.fillna(0)
        # Clipping likelihood values between 0 and 1
        lhoods_idx = pd.IndexSlice[:, :, :, CoordsCols.LIKELIHOOD.value]
        df.loc[:, lhoods_idx] = df.loc[:, lhoods_idx].clip(0, 1)
        # Writing the file
        KeypointsDf.write(df, dst_dir / f"{name}.{KeypointsDf.IO}")
        logger.info("Outputted DLC file.")

    else:
        msg = f"Multiple .h5 files found for {name}. Expected only 1."
        logger.warning(msg)
