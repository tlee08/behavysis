"""Functions have the following format:

Parameters
----------
configs_fp : str
    The experiment config filepath.
default_configs_fp : str
    The default configs filepath to use.
overwrite : str, optional
    How to update the config files

Returns:
-------
str
    Description of the function's outcome.
"""

import logging
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from behavysis.models.experiment_configs import ExperimentConfigs

logger = logging.getLogger(__name__)


class UpdateConfigs:
    """Configuration file management for experiments."""

    @staticmethod
    def update_configs(
        configs_fp: Path,
        default_configs_fp: Path,
        overwrite: Literal["user", "all"],
    ) -> None:
        """Initialises the config files with the given `default_configs`.
        The different types of overwriting are:
        - "user": Only the user parameters are updated.
        - "all": All parameters are updated.

        Parameters
        ----------
        configs_fp : str
            The filepath of the existing config file.
        default_configs_fp : str
            The filepath of the default config file to use.
        overwrite : Literal["user", "all"]
            Specifies how to update the config files.

        Returns:
        -------
        str
            Description of the function's outcome.
        """
        # Parsing in the experiment's existing JSON configs
        try:
            configs = ExperimentConfigs.model_validate_json(configs_fp.read_text())
        except (FileNotFoundError, ValidationError):
            configs = ExperimentConfigs()
        # Reading in the new configs from the given configs_fp
        default_configs = ExperimentConfigs.model_validate_json(
            default_configs_fp.read_text()
        )
        # Overwriting the configs file (with given method)
        if overwrite == "user":
            configs.user = default_configs.user
            configs.ref = default_configs.ref
            logger.info("Updating user and ref configs.")
        elif overwrite == "all":
            configs = default_configs
            logger.info("Updating all configs.")
        else:
            msg = f'Invalid value "{overwrite}" passed to function. The value must be either "user", or "all".'
            raise ValueError(msg)
        # Writing new configs to JSON file
        configs_fp.write_text(configs.model_dump_json(indent=2))
