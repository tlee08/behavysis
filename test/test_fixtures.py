import os

import pytest

from behavysis.utils.io_utils import IOMixin


@pytest.fixture(scope="session", autouse=True)
def proj_dir():
    return os.path.join(".", "tests", "project")


@pytest.fixture(scope="session", autouse=True)
def cleanup(request, proj_dir):
    # Setup: code here will run before your tests

    yield  # this is where the testing happens

    # Teardown
    for i in [
        "0_configs",
        "2_formatted_vid",
        "3_dlc",
        "4_preprocessed",
        "5_features_extracted",
        "6_predicted_behavs",
        "7_scored_behavs",
        "8_analysis",
        "diagnostics",
        "evaluate",
        ".temp",
    ]:
        IOMixin.silent_rm(os.path.join(proj_dir, i))
