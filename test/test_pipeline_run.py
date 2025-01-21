# import os
# import shutil

# import pytest
# from behavysis_pipeline.mixins.io_mixin import IOMixin

# from behavysis_pipeline import Project
# from behavysis_pipeline.processes import *

# from .test_fixtures import cleanup, proj_dir


# def test_pipeline_run(proj_dir):
#     overwrite = True

#     proj = Project(proj_dir)
#     proj.import_experiments()

#     proj.update_configs(
#         default_configs_fp=os.path.join(proj_dir, "default.json"),
#         overwrite="user",
#     )

#     proj.format_vid(
#         (
#             FormatVid.format_vid,
#             FormatVid.get_vid_metadata,
#         ),
#         overwrite=overwrite,
#     )

#     proj.nprocs = 1
#     proj.run_dlc(
#         gputouse=None,
#         overwrite=overwrite,
#     )
#     proj.nprocs = 4

#     proj.calculate_params(
#         (
#             CalculateParams.start_frame,
#             CalculateParams.stop_frame,
#             CalculateParams.px_per_mm,
#         )
#     )

#     proj.preprocess(
#         (
#             Preprocess.start_stop_trim,
#             Preprocess.interpolate,
#             Preprocess.refine_ids,
#         ),
#         overwrite=overwrite,
#     )

#     proj.analyse(
#         (
#             Analyse.thigmotaxis,
#             Analyse.center_crossing,
#             Analyse.speed,
#             Analyse.social_distance,
#             Analyse.freezing,
#         )
#     )

#     proj.combine_analysis_binned()

#     proj.combine_analysis_summary()

#     # proj.extract_features(True, True)
#     # proj.classify_behaviours(True)
#     # proj.export_behaviours(True)
#     # proj.export2csv("7_scored_behavs", "./scored_csv")

#     proj.evaluate(
#         (
#             Evaluate.eval_vid,
#             Evaluate.keypoints_plot,
#         ),
#         overwrite=overwrite,
#     )


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
