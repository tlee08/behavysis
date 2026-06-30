"""Feature extraction from preprocessed keypoints using SimBA."""

import os
import subprocess
import tempfile
from pathlib import Path

import polars as pl
from loguru import logger

from behavysis.constants import CACHE_DIR
from behavysis.models import ExperimentConfig
from behavysis.schemas import KEYPOINTS_SCHEMA, check_bpts_exist, read_df, write_df
from behavysis.utils.io_utils import file_exists_msg
from behavysis.utils.template_utils import save_template

# Order of bodyparts is from
# - https://github.com/sgoldenlab/simba/blob/master/docs/Multi_animal_pose.md
# - https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md
# - https://github.com/sgoldenlab/simba/blob/master/simba/pose_configurations/bp_names/bp_names.csv
# - https://github.com/sgoldenlab/simba/blob/master/simba/pose_configurations/configuration_names/pose_config_names.csv
# 2 animals; 16 body-parts

#####################################################################
#               FEATURE EXTRACTION FOR SIMBA
#####################################################################


def extract_features(
    keypoints_fp: Path,
    features_fp: Path,
    config_fp: Path,
    *,
    overwrite: bool,
) -> None:
    """Extracting features from keypoints dataframe using SimBA processes.

    Parameters
    ----------
    keypoints_fp : Path
        Preprocessed keypoints filepath.
    features_fp : Path
        Filepath to save extracted_features dataframe.
    config_fp : Path
        Config JSON filepath.
    overwrite : bool
        Whether to overwrite the features_fp file (if it exists).

    Returns:
    -------
    str
        The outcome of the process.
    """
    if not overwrite and features_fp.exists():
        logger.warning(file_exists_msg(features_fp))
        return
    name = keypoints_fp.stem
    config_dir = config_fp.parent
    with tempfile.TemporaryDirectory(dir=CACHE_DIR) as _out_dir:
        out_dir = Path(_out_dir)
        simba_in_dir = out_dir / "input"
        simba_dir = out_dir / "simba_proj"
        simba_features_dir = simba_dir / "project_folder" / "csv" / "features_extracted"
        simba_features_fp = simba_features_dir / f"{name}.csv"

        simba_in_dir.mkdir(parents=True, exist_ok=True)
        simba_in_fp = simba_in_dir / f"{name}.csv"

        # Read Polars long-form keypoints
        keypoints_df = read_df(keypoints_fp, KEYPOINTS_SCHEMA)

        # Select bodyparts for SimBA, get frame index, save as CSV
        keypoints_df, index = _select_cols(keypoints_df, config_fp)
        # Reset index (SimBA expects no index column)
        keypoints_df = keypoints_df.drop("frame")
        keypoints_df.write_csv(simba_in_fp)

        # Running SimBA env and script to run SimBA feature extraction
        _run_simba_subproc(simba_dir, simba_in_dir, config_dir, out_dir)

        # Export SimBA feature extraction CSV to disk (re-attach frame index)
        _export2df(simba_features_fp, features_fp, index.to_list())


def _select_cols(
    keypoints_df: pl.DataFrame,
    config_fp: Path,
) -> tuple[pl.DataFrame, pl.Series]:
    """Select given keypoints columns to input to SimBA, output as wide CSV format.

    Pivots long-form to wide (one row per frame, columns = indiv_bpt_coord)
    for SimBA CSV import compatibility.

    Returns (wide DataFrame, frame_index_series).
    """
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    config_filt = config.user.extract_features
    indivs = config.get_ref(config_filt.individuals)
    bpts = config.get_ref(config_filt.bodyparts)

    check_bpts_exist(keypoints_df, bpts)

    # Filter to selected individuals and bodyparts
    filtered = keypoints_df.filter(
        pl.col("individual").is_in(indivs),
        pl.col("bodypart").is_in(bpts),
    )

    # Pivot to wide: one row per frame, columns = "indiv_bpt_coord"
    wide = filtered.select(
        ["frame", "individual", "bodypart", "x", "y", "likelihood"],
    ).sort(["frame", "individual", "bodypart"])

    # Build flat column names: "individual_bodypart_coord"
    wide = wide.with_columns(
        (pl.col("individual") + "_" + pl.col("bodypart") + "_x").alias("col_x"),
        (pl.col("individual") + "_" + pl.col("bodypart") + "_y").alias("col_y"),
        (pl.col("individual") + "_" + pl.col("bodypart") + "_likelihood").alias(
            "col_l",
        ),
    )

    # Pivot x, y, likelihood to columns
    x_cols = wide.select(["frame", "col_x", "x"]).pivot(
        index="frame",
        on="col_x",
        values="x",
    )
    y_cols = wide.select(["frame", "col_y", "y"]).pivot(
        index="frame",
        on="col_y",
        values="y",
    )
    l_cols = wide.select(["frame", "col_l", "likelihood"]).pivot(
        index="frame",
        on="col_l",
        values="likelihood",
    )

    # Save frame index before joining
    frame_index = x_cols.select("frame").to_series()

    # Combine all pivoted columns
    # TODO: confirm
    # result = x_cols.join(y_cols.drop("frame"), how="horizontal").join(
    #     l_cols.drop("frame"),
    #     how="horizontal",
    # )
    result = pl.concat([x_cols, y_cols, l_cols], how="horizontal").drop("frame")

    return result, frame_index


def _run_simba_subproc(
    simba_dir: Path,
    keypoints_dir: Path,
    config_dir: Path,
    temp_dir: Path,
) -> None:
    """Running SimBA script to feature engineer from x-y-likelihood pts.

    A custom SimBA script must be run in a
    separate custom conda environment because SimBA
    cannot be installed in the same environment
    as DEEPLABCUT (and also uses Python 3.6 - which is old).
    """
    # Saving the script to a file
    script_fp = temp_dir / "simba_subproc.py"
    save_template(
        "simba_subproc.py",
        script_fp,
        simba_dir=simba_dir,
        keypoints_dir=keypoints_dir,
        config_dir=config_dir,
    )
    # Running the Simba subprocess in a separate conda env
    cmd = [
        os.environ["CONDA_EXE"],
        "run",
        "--no-capture-output",
        "-n",
        "simba",
        "python",
        str(script_fp),
    ]
    subprocess.run(cmd, check=True)


def _export2df(in_fp: Path, dst_fp: Path, index: list[int]) -> None:
    """Export SimBA features CSV to Polars parquet, re-attaching frame index."""
    features_df = pl.read_csv(in_fp)

    # Re-attach frame index (lost during SimBA CSV roundtrip)
    features_df = features_df.with_columns(
        pl.Series("frame", index, dtype=pl.Int64),
    )

    # Move frame to first column
    cols = ["frame"] + [c for c in features_df.columns if c != "frame"]
    features_df = features_df.select(cols)

    write_df(
        features_df,
        dst_fp,
        {},
    )  # dynamic schema — validated by read_df at consumer
    logger.info("Exported SimBA features to disk.")
