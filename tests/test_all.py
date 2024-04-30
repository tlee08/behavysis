import os
import shutil

import pytest

from behavysis_pipeline import Project
from behavysis_pipeline.processes import *


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
        "analysis",
        "diagnostics",
        "evaluate",
        ".temp",
    ]:
        if os.path.exists(os.path.join(proj_dir, i)):
            shutil.rmtree(os.path.join(proj_dir, i))


def test_BehavysisProject(proj_dir):
    proj = Project(proj_dir)
    assert isinstance(proj, Project)


def test_import_experiments(proj_dir):
    proj = Project(proj_dir)
    proj.import_experiments()
    assert len(proj.get_experiments()) > 0


def test_update_configs(proj_dir):
    proj = Project(proj_dir)
    proj.update_configs(
        default_configs_fp=os.path.join(proj_dir, "default.json"),
        overwrite="user",
    )
    # Add assertions here based on what you expect to happen


# def test_format_vid():
#     proj_dir = os.path.join(".")
#     proj = BehavysisProject(proj_dir)
#     proj.format_vid(
#         funcs=(
#             FormatVid.format_vid,
#             FormatVid.vid_metadata,
#         ),
#         overwrite=True,
#     )
#     # Add assertions here based on what you expect to happen


# def test_run_dlc():
#     proj_dir = os.path.join(".")
#     proj = BehavysisProject(proj_dir)
#     proj.run_dlc(
#         gputouse=None,
#         overwrite=True,
#     )
#     # Add assertions here based on what you expect to happen


# def test_calculate_params():
#     proj_dir = os.path.join(".")
#     proj = BehavysisProject(proj_dir)
#     proj.calculate_params(
#         (
#             CalculateParams.start_frame,
#             CalculateParams.stop_frame,
#             CalculateParams.px_per_mm,
#         )
#     )
#     # Add assertions here based on what you expect to happen


# def test_preprocess():
#     proj_dir = os.path.join(".")
#     proj = BehavysisProject(proj_dir)
#     proj.preprocess(
#         (
#             Preprocess.start_stop_trim,
#             Preprocess.interpolate,
#             Preprocess.refine_ids,
#         ),
#         overwrite=True,
#     )
#     # Add assertions here based on what you expect to happen
