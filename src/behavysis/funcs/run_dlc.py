"""Functions have the following format."""

import os
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import polars as pl
from loguru import logger

from behavysis.constants import (
    BODYPART,
    CACHE_DIR,
    DF_IO_FORMAT,
    FRAME,
    INDIVIDUAL,
    LIKELIHOOD,
    X,
    Y,
)
from behavysis.models import ExperimentConfig
from behavysis.schemas import KEYPOINTS_SCHEMA, write_df
from behavysis.utils.template_utils import save_template

DLC_HDF_KEY = "data"


def ma_dlc_run_single(
    vid_fp: Path,
    keypoints_dir: Path,
    config: ExperimentConfig,
    gputouse: int | None,
) -> None:
    """Running DLC script to generate a keypoints dataframe from a single video."""
    # Derive more parameters
    with tempfile.TemporaryDirectory(dir=CACHE_DIR) as _out_dir:
        out_dir = Path(_out_dir)
        # Running the DLC subprocess (in a separate conda env)
        _run_dlc_subproc(
            config.require_run_dlc().model_fp,
            [vid_fp],
            out_dir,
            gputouse,
        )
        # Exporting the h5 to chosen file format
        _export2df(vid_fp.stem, out_dir, keypoints_dir)


def ma_dlc_run_batch(
    vid_fp_ls: list[Path],
    keypoints_dir: Path,
    dlc_config_fp: Path,
    gputouse: int | None,
) -> None:
    """Running DLC to generate a keypoints dataframe from a single video."""
    # If there are no videos to process, return
    if len(vid_fp_ls) == 0:
        return
    with tempfile.TemporaryDirectory(dir=CACHE_DIR) as _out_dir:
        out_dir = Path(_out_dir)
        # Running the DLC subprocess (in a separate conda env)
        _run_dlc_subproc(dlc_config_fp, vid_fp_ls, out_dir, gputouse)
        # Exporting the h5 to chosen file format
        for vid_fp in vid_fp_ls:
            _export2df(vid_fp.stem, out_dir, keypoints_dir)


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
    subprocess.run(cmd, check=True)


def _export2df(name: str, src_dir: Path, dst_dir: Path) -> None:
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
        logger.warning(msg)
        return
    if len(name_fp_ls) != 1:
        msg = f"Multiple .h5 files found for {name}. Expected only 1."
        logger.warning(msg)
        return

    # Get the only value in the 1-element list
    name_fp = src_dir / name_fp_ls[0]
    # Read h5 as pandas (DLC outputs pandas MultiIndex columns)
    df_pd = pd.DataFrame(pd.read_hdf(name_fp))
    # Impute na values with 0
    df_pd = df_pd.fillna(0)
    # Drop scorer level (always single value, useless)
    df_pd.columns = df_pd.columns.droplevel("scorer")
    # Name index as "frame"
    df_pd.index.name = FRAME
    # Stack bodyparts + individuals + coords into rows, then unstack coords
    # to get x, y, likelihood as columns
    long_df = (
        df_pd.stack(["individuals", "bodyparts", "coords"])  # noqa: PD010, PD013
        .unstack("coords")
        .reset_index()
    )
    # Convert to Polars long form
    df_pl = pl.from_pandas(long_df.reset_index()).select(
        pl.col(FRAME).cast(pl.Int64),
        pl.col("individuals").alias(INDIVIDUAL),
        pl.col("bodyparts").alias(BODYPART),
        pl.col(X).cast(pl.Float64),
        pl.col(Y).cast(pl.Float64),
        pl.col(LIKELIHOOD).cast(pl.Float64),
    )
    # Write to file
    write_df(df_pl, dst_dir / f"{name}.{DF_IO_FORMAT}", KEYPOINTS_SCHEMA)
    logger.info("Outputted DLC file.")
