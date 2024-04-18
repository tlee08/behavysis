"""
Utility functions.
"""

from __future__ import annotations

import os
import re
from inspect import currentframe
from multiprocessing import current_process
from subprocess import PIPE, Popen
from typing import Callable, Optional, Type, TypeVar, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel

from ba_package.utils.constants import (
    DIAGNOSTICS_SUCCESS_MESSAGES,
    DLC_COLUMN_NAMES,
    HDF_KEY,
    PROCESS_COL,
    PROCS,
    SINGLE_COL,
    TEMP_DIR,
)

#####################################################################
#               CONFIG FILE HELPER FUNCS
#####################################################################


T = TypeVar("T", bound=BaseModel)


def read_configs(fp: str, model_class: Type[T]) -> T:
    """
    Returns the config model from the specified JSON config file.

    Parameters
    ----------
    fp : str
        Filepath of the JSON config file.
    model_class : Type[BaseModel]
        The BaseModel class for type hints.

    Returns
    -------
    Type[BaseModel]
        The config model.

    Notes
    -----
    This function reads the contents of the JSON config file located at `fp` and
    returns the config model. The `model_class` parameter is used for type hints.

    Example
    -------
    >>> config = read_configs("/path/to/config.json", ConfigModel)
    """
    with open(fp, "r", encoding="utf-8") as f:
        return model_class.model_validate_json(f.read())


def write_configs(configs: BaseModel, fp: str) -> None:
    """
    Writes the given configs model to the configs file (i.e. hence updating the file)

    Parameters
    ----------
    configs : Configs
        Configs model to write to file.
    fp : str
        File to save configs to.
    """
    os.makedirs(os.path.split(fp)[0], exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        f.write(configs.model_dump_json(indent=2))


#####################################################################
#           DATA FRAME READER/WRITER FUNCS (CSV, H5, Feather)
#####################################################################


def read_dlc_csv(fp: str) -> pd.DataFrame:
    """
    Reading in DLC csv file.

    Parameters
    ----------
    fp : str
        _description_

    Returns
    -------
    pd.DataFrame
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    try:
        return pd.read_csv(
            fp, header=np.arange(len(DLC_COLUMN_NAMES)).tolist(), index_col=0
        ).sort_index()
    except Exception as e:
        raise ValueError(
            f'The csv file, "{fp}", does not exist or is in an invalid format.'
            + "Please check this file."
        ) from e


def write_dlc_csv(df: pd.DataFrame, fp: str) -> None:
    """
    Writing DLC dataframe to csv file.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    fp : str
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    os.makedirs(os.path.split(fp)[0], exist_ok=True)
    try:
        df.to_csv(fp)
    except Exception as e:
        raise ValueError(e) from e


def read_h5(fp: str) -> pd.DataFrame:
    """
    Reading h5 file.

    Parameters
    ----------
    fp : str
        _description_

    Returns
    -------
    pd.DataFrame
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    try:
        return pd.DataFrame(pd.read_hdf(fp, key=HDF_KEY, mode="r").sort_index())
    except Exception as e:
        raise ValueError(
            f'The h5 file, "{fp}", does not exist or is in an invalid format.'
            + "Please check this file."
        ) from e


def write_h5(df: pd.DataFrame, fp: str) -> None:
    """
    Writing dataframe h5 file.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    fp : str
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    os.makedirs(os.path.split(fp)[0], exist_ok=True)
    try:
        df.to_hdf(fp, key=HDF_KEY, mode="w")
    except Exception as e:
        raise ValueError(e) from e


def read_feather(fp: str) -> pd.DataFrame:
    """
    Reading feather file.

    Parameters
    ----------
    fp : str
        _description_

    Returns
    -------
    pd.DataFrame
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    try:
        return pd.read_feather(fp).sort_index()
    except Exception as e:
        raise ValueError(
            f'The feather file, "{fp}", does not exist or is in an invalid format.'
            + "Please check this file."
        ) from e


def write_feather(df: pd.DataFrame, fp: str) -> None:
    """
    Writing dataframe feather file.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    fp : str
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    os.makedirs(os.path.split(fp)[0], exist_ok=True)
    try:
        df.to_feather(fp)
    except Exception as e:
        raise ValueError(e) from e


#####################################################################
#               MISC FUNCS
#####################################################################


def success_msg() -> str:
    """
    Return a random positive message :)

    Returns
    -------
    str
        _description_
    """
    return np.random.choice(DIAGNOSTICS_SUCCESS_MESSAGES)


def check_bpts_exist(bodyparts: list, dlc_df: pd.DataFrame) -> None:
    """
    _summary_

    Parameters
    ----------
    bodyparts : list
        _description_
    dlc_df : pd.DataFrame
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    # Checking that the bodyparts are all valid:
    bodyparts_exist = np.isin(bodyparts, dlc_df.columns.unique("bodyparts"))
    if not bodyparts_exist.all():
        msg = (
            "Some bodyparts in the config file are missing from the csv file.\n"
            + "They are:\n"
        )
        for bp in np.array(bodyparts)[~bodyparts_exist]:
            msg += f"    - {bp}\n"
        raise ValueError(msg)


def warning_msg(func: Optional[Callable] = None):
    """
    Return a warning message for the given function.

    Parameters
    ----------
    func : Callable, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    if not func:
        func = currentframe().f_back.f_code.co_name
    return (
        "WARNING: Output file already exists - not overwriting file.\n"
        + "To overwrite, specify {}(..., overwrite=True).\n"
    ).format(func)


def get_dlc_headings(dlc_df: pd.DataFrame) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Returns a tuple of the individuals (animals, not "single"), and tuple of the multi-animal
    bodyparts.

    Parameters
    ----------
    dlc_df : pd.DataFrame
        DLC pd.DataFrame.

    Returns
    -------
    tuple[tuple[str, ...], tuple[str, ...]]
        `(indivs_ls, bpts_ls)` tuples. It is recommended to unpack these vals.
    """
    # Getting DLC column MultiIndex
    columns = dlc_df.columns
    # Filtering out any single and processing columns
    # Getting individuals to filter out
    filt_cols = [PROCESS_COL, SINGLE_COL]
    # Filtering out
    for filt_col in filt_cols:
        if filt_col in columns.unique("individuals"):
            columns = columns.drop(filt_col, level="individuals")
    # Getting individuals list
    indivs = columns.unique("individuals").to_list()
    # Getting bodyparts list
    bpts = columns.unique("bodyparts").to_list()
    return indivs, bpts


def clean_dlc_headings(dlc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops the "scorer" level (and any other unnecessary levels) in the column
    header of the dataframe. This makes analysis easier.

    Parameters
    ----------
    dlc_df : pd.DataFrame
        DLC pd.DataFrame.

    Returns
    -------
    pd.DataFrame
        DLC pd.DataFrame.
    """
    dlc_df = dlc_df.copy()
    # Removing the scorer column because all values are identical
    dlc_df.columns = dlc_df.columns.droplevel("scorer")
    # Grouping the columns by the individuals level for cleaner presentation
    dlc_df = dlc_df.reindex(
        columns=dlc_df.columns.unique("individuals"), level="individuals"
    )
    return dlc_df


def vect_to_bouts(vect: Union[np.ndarray, pd.Series]) -> pd.DataFrame:
    """
    Will return a dataframe with the start and stop indexes of each contiguous set of
    positive values (i.e. a bout).

    Parameters
    ----------
    vect : Union[np.ndarray, pd.Series]
        Expects a vector of booleans

    Returns
    -------
    pd.DataFrame
        _description_
    """
    offset = 0
    if isinstance(vect, pd.Series):
        offset = vect.index[0]
    # Getting stop and start indexes of each bout
    z = np.concatenate(([0], vect, [0]))
    start = np.flatnonzero(~z[:-1] & z[1:])
    stop = np.flatnonzero(z[:-1] & ~z[1:]) - 1
    bouts_ls = np.column_stack((start, stop)) + offset
    # Making dataframe
    bouts_df = pd.DataFrame(bouts_ls, columns=["start", "stop"])
    bouts_df["dur"] = bouts_df["stop"] - bouts_df["start"]
    return bouts_df


#####################################################################
#               DIR HELPER FUNCS
#####################################################################


def clear_dir_junk(my_dir: str) -> None:
    """
    Removes all hidden files in given directory.
    Hidden files begin with ".".

    Parameters
    ----------
    my_dir : str
        Directory to clear.
    """
    for i in os.listdir(dir):
        path = os.path.join(my_dir, i)
        if re.search(r"^\.", i):
            os.remove(path)


def silent_remove(fp: str) -> None:
    """
    Removes the given file if it exists.
    Does nothing if not.
    Does not throw any errors,

    Parameters
    ----------
    fp : str
        Filepath to remove.
    """
    try:
        os.remove(fp)
    except OSError:
        pass


def get_name(fp: str) -> str:
    """
    Given the filepath, returns the name of the file.
    The name is:
    ```
    <path_to_file>/<name>.<ext>
    ```

    Parameters
    ----------
    fp : str
        Filepath.

    Returns
    -------
    str
        File name.
    """
    return os.path.splitext(os.path.split(fp)[1])[0]


#####################################################################
#           SUBPROCESS AND MULTIPROCESS HELPER FUNCS
#####################################################################


def get_cpid() -> int:
    """Get child process ID for multiprocessing."""
    # Mod is very hacky, but this works for now
    return current_process()._identity[0] % PROCS if current_process()._identity else 0


def run_subprocess_fstream(cmd: list[str], fp: str = None) -> None:
    """Run a subprocess and stream the output to a file."""
    if not fp:
        fp = os.path.join(TEMP_DIR, "subprocess_output.txt")
    # Making a file to store the output
    os.makedirs(os.path.split(fp)[0], exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        # Starting the subprocess
        with Popen(cmd, stdout=f, stderr=f) as p:
            # Wait for the subprocess to finish
            p.wait()
            # Error handling
            if p.returncode:
                raise ValueError(p.stderr.read().decode())


def run_subprocess_str(cmd: list[str]) -> str:
    """Run a subprocess and return the output as a string."""
    # Running the subprocess
    with Popen(cmd, stdout=PIPE, stderr=PIPE) as p:
        # Wait for the subprocess to finish
        out, err = p.communicate()
        # Error handling
        if p.returncode:
            raise ValueError(err.decode("utf-8"))
        return out.decode("utf-8")
