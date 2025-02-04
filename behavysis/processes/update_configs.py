"""
Functions have the following format:

Parameters
----------
configs_fp : str
    The experiment config filepath.
default_configs_fp : str
    The default configs filepath to use.
overwrite : str, optional
    How to update the config files

Returns
-------
str
    Description of the function's outcome.
"""

from typing import Literal

from pydantic import ValidationError

from behavysis.pydantic_models.experiment_configs import ExperimentConfigs
from behavysis.utils.logging_utils import get_io_obj_content, init_logger_io_obj


class UpdateConfigs:
    """_summary_"""

    @staticmethod
    def update_configs(
        configs_fp: str,
        default_configs_fp: str,
        overwrite: Literal["user", "all"],
    ) -> str:
        """
        Initialises the config files with the given `default_configs`.
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

        Returns
        -------
        str
            Description of the function's outcome.
        """
        logger, io_obj = init_logger_io_obj()
        # Parsing in the experiment's existing JSON configs
        try:
            configs = ExperimentConfigs.read_json(configs_fp)
        except (FileNotFoundError, ValidationError):
            configs = ExperimentConfigs()
        # Reading in the new configs from the given configs_fp
        default_configs = ExperimentConfigs.read_json(default_configs_fp)
        # Overwriting the configs file (with given method)
        if overwrite == "user":
            configs.user = default_configs.user
            configs.ref = default_configs.ref
            logger.info("Updating user and ref configs.")
        elif overwrite == "all":
            configs = default_configs
            logger.info("Updating all configs.")
        else:
            raise ValueError(
                f'Invalid value "{overwrite}" passed to function. ' 'The value must be either "user", or "all".'
            )
        # Writing new configs to JSON file
        configs.write_json(configs_fp)
        return get_io_obj_content(io_obj)
