"""Functions have the following format."""

import os
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import polars as pl
from loguru import logger

from behavysis.constants import CACHE_DIR, DF_IO_FORMAT, LIKELIHOOD
from behavysis.models import ExperimentConfig
from behavysis.schemas import KEYPOINTS_SCHEMA, write_df
from behavysis.utils.io_utils import file_exists_msg, silent_remove
from behavysis.utils.template_utils import save_template

DLC_HDF_KEY = "data"


def ma_dlc_run_single(
    vid_fp: Path,
    keypoints_fp: Path,
    config_fp: Path,
    gputouse: int | None,
    *,
    overwrite: bool,
) -> None:
    """Running DLC script to generate a keypoints dataframe from a single video."""
    if not overwrite and keypoints_fp.exists():
        logger.warning(file_exists_msg(keypoints_fp))
        return
    # Getting model_fp
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    model_fp = config.get_ref(config.user.run_dlc.model_fp)
    # Derive more parameters
    keypoints_dir = keypoints_fp.parent

    with tempfile.TemporaryDirectory(dir=CACHE_DIR) as _out_dir:
        out_dir = Path(_out_dir)
        # Running the DLC subprocess (in a separate conda env)
        _run_dlc_subproc(
            model_fp,
            [vid_fp],
            out_dir,
            CACHE_DIR,
            gputouse,
        )
        # Exporting the h5 to chosen file format
        _export2df(vid_fp.stem, out_dir, keypoints_dir)


def ma_dlc_run_batch(
    vid_fp_ls: list[Path],
    keypoints_dir: Path,
    config_dir: Path,
    gputouse: int | None,
    *,
    overwrite: bool,
) -> None:
    """Running DLC to generate a keypoints dataframe from a single video."""
    # If overwrite is False, filtering for only experiments that need processing
    if not overwrite:
        # Getting only the vid_fp_ls elements that do not exist in keypoints_dir
        vid_fp_ls = [
            vid_fp
            for vid_fp in vid_fp_ls
            if not (keypoints_dir / f"{vid_fp.stem}.{DF_IO_FORMAT}").exists()
        ]

    # If there are no videos to process, return
    if len(vid_fp_ls) == 0:
        return

    # Getting the DLC model config path
    # Getting the names of the files that need processing
    dlc_fp_ls = [i.stem for i in vid_fp_ls]
    # Getting their corresponding config_fp
    dlc_fp_ls = [config_dir / f"{i}.json" for i in dlc_fp_ls]
    # Reading their config
    dlc_fp_ls = [ExperimentConfig.model_validate_json(i.read_text()) for i in dlc_fp_ls]
    # Getting their model_fp
    dlc_fp_ls = [i.user.run_dlc.model_fp for i in dlc_fp_ls]
    # Converting to a set
    dlc_fp_set = set(dlc_fp_ls)
    # Assertion: all model_fp must be the same
    assert len(dlc_fp_set) == 1
    # Getting the model_fp
    model_fp = dlc_fp_set.pop()

    with tempfile.TemporaryDirectory(dir=CACHE_DIR) as _out_dir:
        out_dir = Path(_out_dir)
        # Running the DLC subprocess (in a separate conda env)
        _run_dlc_subproc(model_fp, vid_fp_ls, out_dir, CACHE_DIR, gputouse)
        # Exporting the h5 to chosen file format
        for vid_fp in vid_fp_ls:
            _export2df(vid_fp.stem, out_dir, keypoints_dir)


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
        script_fp,
        vid_fp_ls=[str(_i) for _i in vid_fp_ls],
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

    name_fp = src_dir / name_fp_ls[0]
    # Read h5 as pandas (DLC outputs pandas MultiIndex columns)
    df_pd = pd.DataFrame(pd.read_hdf(name_fp))
    df_pd = df_pd.fillna(0)

    # Clip likelihood values between 0 and 1
    lhoods_idx = pd.IndexSlice[:, :, :, LIKELIHOOD]
    df_pd.loc[:, lhoods_idx] = df_pd.loc[:, lhoods_idx].clip(0, 1)

    # Drop scorer level (always single value, useless)
    df_pd.columns = df_pd.columns.droplevel("scorer")

    # Stack bodyparts + individuals + coords into rows, then unstack coords
    # to get x, y, likelihood as columns
    stacked = df_pd.melt(["individuals", "bodyparts", "coords"])
    unstacked = stacked.unstack("coords")
    unstacked.columns = unstacked.columns.str.lower()

    # Convert to Polars long form
    df_pl = pl.from_pandas(unstacked.reset_index()).select(
        pl.col("frame").cast(pl.Int64),
        pl.col("individuals").alias("individual"),
        pl.col("bodyparts").alias("bodypart"),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("likelihood").cast(pl.Float64),
    )

    write_df(df_pl, dst_dir / f"{name}.{DF_IO_FORMAT}", KEYPOINTS_SCHEMA)
    logger.info("Outputted DLC file.")
