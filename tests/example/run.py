"""_summary_"""

import os
import shutil
import sys

from behavysis_pipeline import *
from behavysis_pipeline.processes import *

if __name__ == "__main__":
    overwrite = True

    proj_dir = os.path.join(".")

    proj = Project(proj_dir)
    proj.import_experiments()
    # proj = proj.get_experiments()[0]

    # proj.nprocs = 4

    default_configs_fp = os.path.join(proj_dir, "default.json")

    proj.update_configs(
        default_configs_fp,
        overwrite="user",
    )
    proj.format_vid(
        (
            # FormatVid.format_vid,
            FormatVid.get_vid_metadata,
        ),
        overwrite=overwrite,
    )
    # proj.run_dlc(
    #     gputouse=None,
    #     overwrite=overwrite,
    # )
    # proj.calculate_params(
    #     (
    #         CalculateParams.start_frame,
    #         CalculateParams.stop_frame,
    #         CalculateParams.px_per_mm,
    #     )
    # )
    # proj.preprocess(
    #     (
    #         Preprocess.start_stop_trim,
    #         Preprocess.interpolate,
    #         Preprocess.refine_ids,
    #     ),
    #     overwrite=overwrite,
    # )
    # proj.extract_features(True)
    # proj.classify_behaviours(True)
    # proj.export_behaviours(True)
    # # proj.export_feather("7_scored_behavs", "./scored_csv")
    # proj.analyse(
    #     (
    #         Analyse.thigmotaxis,
    #         Analyse.center_crossing,
    #         Analyse.in_roi,
    #         Analyse.speed,
    #         Analyse.social_distance,
    #         Analyse.freezing,
    #     )
    # )
    # proj.collate_configs_auto()
    # proj.collate_analysis_binned()
    # proj.collate_analysis_summary()
    # proj.evaluate(
    #     (
    #         Evaluate.eval_vid,
    #         Evaluate.keypoints_plot,
    #     ),
    #     overwrite=overwrite,
    # )

    # # import shutil
    # # import os

    # # for i in [
    # #     "0_configs",
    # #     "4_preprocessed",
    # #     "5_features_extracted",
    # #     "6_predicted_behavs",
    # #     "7_scored_behavs",
    # #     "8_analysis",
    # #     "diagnostics",
    # #     "evaluate",
    # # ]:
    # #     if os.path.exists(os.path.join(proj_dir, i)):
    # #         shutil.rmtree(os.path.join(proj_dir, i))
