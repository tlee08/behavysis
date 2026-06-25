"""Functions have the following format."""

from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import ValidationError

from behavysis.models import ExperimentConfig


def update_config(
    config_fp: Path,
    default_config_fp: Path,
    overwrite: Literal["user", "all"],
) -> None:
    """Initialises the config files with the given `default_config`.

    The different types of overwriting are:
    - "user": Only the user parameters are updated.
    - "all": All parameters are updated.

    Parameters
    ----------
    config_fp : str
        The filepath of the existing config file.
    default_config_fp : str
        The filepath of the default config file to use.
    overwrite : Literal["user", "all"]
        Specifies how to update the config files.

    Returns:
    -------
    str
        Description of the function's outcome.
    """
    # Parsing in the experiment's existing JSON config
    try:
        config = ExperimentConfig.model_validate_json(config_fp.read_text())
    except (FileNotFoundError, ValidationError):
        config = ExperimentConfig()
    # Reading in the new config from the given config_fp
    default_config = ExperimentConfig.model_validate_json(default_config_fp.read_text())
    # Overwriting the config file (with given method)
    if overwrite == "user":
        config.user = default_config.user
        config.ref = default_config.ref
        logger.info("Updating user and ref config.")
    elif overwrite == "all":
        config = default_config
        logger.info("Updating all config.")
    else:
        msg = f'Invalid overwrite value: "{overwrite}"\n  Expected: "user" or "all"'
        raise ValueError(msg)
    # Writing new config to JSON file
    config_fp.write_text(config.model_dump_json(indent=2))
