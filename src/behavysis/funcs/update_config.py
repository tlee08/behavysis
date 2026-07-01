"""Functions have the following format."""

import shutil
from pathlib import Path

from behavysis.models import ExperimentConfig


def update_config(
    config_fp: Path,
    default_config_fp: Path,
) -> None:
    """Initialises the config files with the given `default_config`."""
    # Parsing in the new config to see if it is valid
    ExperimentConfig.read_yaml(default_config_fp)
    # Overwriting the config file with the new config
    shutil.copyfile(default_config_fp, config_fp)
