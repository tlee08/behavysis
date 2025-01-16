# import os
# import shutil

# import pytest
# from behavysis_pipeline.mixins.io_mixin import IOMixin

# from .test_fixtures import cleanup, proj_dir

# from behavysis_pipeline import Project
# from behavysis_pipeline.processes import *


# def test_Project(proj_dir):
#     proj = Project(proj_dir)
#     assert isinstance(proj, Project)


# def test_import_experiments(proj_dir):
#     proj = Project(proj_dir)
#     proj.import_experiments()
#     assert len(proj.get_experiments()) > 0


# def test_update_configs(proj_dir):
#     proj = Project(proj_dir)
#     proj.update_configs(
#         default_configs_fp=os.path.join(proj_dir, "default.json"),
#         overwrite="user",
#     )
#     # Add assertions here based on what you expect to happen


# # def test_format_vid():
# #     proj_dir = os.path.join(".")
# #     proj = Project(proj_dir)
# #     proj.format_vid(
# #         funcs=(
# #             FormatVid.format_vid,
# #             FormatVid.vid_metadata,
# #         ),
# #         overwrite=True,
# #     )
# #     # Add assertions here based on what you expect to happen


# # def test_run_dlc():
# #     proj_dir = os.path.join(".")
# #     proj = Project(proj_dir)
# #     proj.run_dlc(
# #         gputouse=None,
# #         overwrite=True,
# #     )
# #     # Add assertions here based on what you expect to happen


# # def test_calculate_params():
# #     proj_dir = os.path.join(".")
# #     proj = Project(proj_dir)
# #     proj.calculate_params(
# #         (
# #             CalculateParams.start_frame,
# #             CalculateParams.stop_frame,
# #             CalculateParams.px_per_mm,
# #         )
# #     )
# #     # Add assertions here based on what you expect to happen


# # def test_preprocess():
# #     proj_dir = os.path.join(".")
# #     proj = Project(proj_dir)
# #     proj.preprocess(
# #         (
# #             Preprocess.start_stop_trim,
# #             Preprocess.interpolate,
# #             Preprocess.refine_ids,
# #         ),
# #         overwrite=True,
# #     )
# #     # Add assertions here based on what you expect to happen
