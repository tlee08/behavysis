"""_summary_"""

import os
import shutil
import sys

from behavysis_pipeline import *
from behavysis_pipeline.processes import *

if __name__ == "__main__":
    overwrite = True

    proj_dir = os.path.join(".")
    proj = BehavysisProject(proj_dir)
    proj.import_experiments()
    # exp = proj.get_experiments()[1]

    # proj.nprocs = 4

    proj.update_configs(
        os.path.join(proj_dir, "default.json"),
        overwrite="all",
    )
    proj.format_vid(
        (
            FormatVid.format_vid,
            FormatVid.get_vid_metadata,
        ),
        overwrite=overwrite,
    )
    proj.run_dlc(
        gputouse=None,
        overwrite=overwrite,
    )
    proj.calculate_params(
        (
            CalculateParams.start_frame,
            CalculateParams.stop_frame,
            CalculateParams.px_per_mm,
        )
    )
    proj.preprocess(
        (
            Preprocess.start_stop_trim,
            Preprocess.interpolate_points,
            Preprocess.bodycentre,
            Preprocess.refine_identities,
        ),
        overwrite=overwrite,
    )
    proj.analyse(
        (
            Analyse.thigmotaxis,
            Analyse.center_crossing,
            Analyse.speed,
            Analyse.social_distance,
            Analyse.freezing,
            Analyse.in_roi,
        )
    )
    proj.combine_analysis_binned()
    proj.combine_analysis_summary()
    proj.extract_features(True, True)
    proj.classify_behaviours(True)
    proj.export_behaviours(True)
    proj.export_feather("7_scored_behavs", "./scored_csv")
    proj.evaluate(
        (
            Evaluate.eval_vid,
            Evaluate.keypoints_plot,
        ),
        overwrite=overwrite,
    )

    # import shutil
    # import os

    # for i in [
    #     "0_configs",
    #     "4_preprocessed",
    #     "5_features_extracted",
    #     "6_predicted_behavs",
    #     "7_scored_behavs",
    #     "analysis",
    #     "diagnostics",
    #     "evaluate",
    # ]:
    #     if os.path.exists(os.path.join(proj_dir, i)):
    #         shutil.rmtree(os.path.join(proj_dir, i))
