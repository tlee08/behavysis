"""Functions have the following format."""

import os
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import polars as pl
from loguru import logger

from behavysis.constants import CACHE_DIR
from behavysis.models import ExperimentConfig
from behavysis.schemas import KEYPOINTS_SCHEMA, write_df
from behavysis.transforms import convert_raw_dlc_to_keypoints
from behavysis.utils import save_template

DLC_HDF_KEY = "data"


def dlc_run_ma(
    vid_fp: Path,
    keypoints_fp: Path,
    config: ExperimentConfig,
    gputouse: int | None,
) -> None:
    """Running DLC script to generate a keypoints dataframe from a single video."""
    # Using a temporary directory to store the DLC output files
    with tempfile.TemporaryDirectory(dir=CACHE_DIR) as _out_dir:
        out_dir = Path(_out_dir)
        # Running the DLC subprocess (in a separate conda env)
        _run_dlc_subproc(config.require_run_dlc().model_fp, [vid_fp], out_dir, gputouse)
        # Converting the h5 to long
        df = _export2df(vid_fp.stem, out_dir)
        # Write to file
        write_df(df, keypoints_fp, KEYPOINTS_SCHEMA)


def _run_dlc_subproc(
    dlc_config_fp: Path,
    vid_fp_ls: list[Path],
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
        "dlc/dlc_subproc.py",
        script_fp,
        vid_fp_ls=[str(_i) for _i in vid_fp_ls],
        model_fp=dlc_config_fp,
        temp_dir=temp_dir,
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
    subprocess.run(cmd, check=True)  # noqa: S603


def _export2df(name: str, src_dir: Path) -> pl.DataFrame:
    """Export DLC h5 output to Polars long-form parquet.

    Reads pandas MultiIndex h5, unstacks to long form, drops the scorer level
    (always nunique=1), converts to Polars, and writes parquet.
    """
    # Get the corresponding .h5 filename
    name_fp_ls = [
        i for i in src_dir.iterdir() if re.search(rf"^{name}DLC.*\.h5$", i.name)
    ]
    if len(name_fp_ls) == 0:
        msg = f"No .h5 file found for {name}."
        raise ValueError(msg)
    if len(name_fp_ls) != 1:
        msg = f"Multiple .h5 files found for {name}. Expected only 1."
        raise ValueError(msg)
    # Get the only value in the 1-element list
    name_fp = src_dir / name_fp_ls[0]
    # Read h5 as pandas (DLC outputs pandas MultiIndex columns)
    df = pd.DataFrame(pd.read_hdf(name_fp))
    # Convert and return
    return convert_raw_dlc_to_keypoints(df)
