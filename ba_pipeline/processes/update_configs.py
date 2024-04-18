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

from typing import Literal, Type

from pydantic import BaseModel, ValidationError

from ba_pipeline.utils.funcs import read_configs, write_configs


class UpdateConfigs:
    """_summary_"""

    @staticmethod
    def update_configs(
        configs_fp: str,
        default_configs_fp: str,
        overwrite: Literal["user", "all"],
        model_class: Type[BaseModel],
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
        model_class : Type[BaseModel]
            The Pydantic model class used for validation.

        Returns
        -------
        str
            Description of the function's outcome.
        """
        outcome = ""
        # Parsing in the experiment's existing JSON configs
        try:
            configs = read_configs(configs_fp, model_class)
        except (FileNotFoundError, ValidationError):
            configs = model_class()
        # Reading in the new configs from the given configs_fp
        default_configs = read_configs(default_configs_fp, model_class)
        # Overwriting the configs file (with given method)
        if overwrite == "user":
            configs.user = default_configs.user
            outcome += "Updating user configs.\n"
        elif overwrite == "all":
            configs = default_configs
            outcome += "Updating all parameters.\n"
        else:
            raise ValueError(
                f'Invalid value "{overwrite}" passed to function. '
                + 'The value must be either "user", or "all".'
            )
        # Writing new configs to JSON file
        write_configs(configs, configs_fp)
        return outcome
