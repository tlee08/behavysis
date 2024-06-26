"""_summary_"""

import os

import pandas as pd

from behavysis_pipeline import Project
from behavysis_pipeline.processes import FormatVid, CalculateParams, Analyse, Evaluate, Preprocess

if __name__ == "__main__":
    overwrite = True

    proj_dir = os.path.join(".")
    proj = Project(proj_dir)
    proj.import_experiments()
    exp = proj.get_experiments()[1]

    proj.nprocs = 5

    default_configs_fp = os.path.join(proj_dir, "default_configs.json")
    proj.update_configs(
        default_configs_fp,
        overwrite="user",
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
            CalculateParams.exp_dur,
            CalculateParams.px_per_mm,
        )
    )

    proj.collate_configs_auto()

    proj.preprocess(
        (
            Preprocess.start_stop_trim,
            Preprocess.interpolate,
            Preprocess.refine_ids,
        ),
        overwrite=overwrite,
    )

    proj.analyse(
        (
            Analyse.thigmotaxis,
            Analyse.center_crossing,
            Analyse.in_roi,
            Analyse.speed,
            Analyse.social_distance,
            Analyse.freezing,
        )
    )
    proj.collate_analysis_binned()
    proj.collate_analysis_summary()

    proj.extract_features(overwrite)
    proj.classify_behaviours(overwrite)
    proj.export_behaviours(overwrite)

    proj.behav_analyse()

    proj.evaluate(
        (
            Evaluate.eval_vid,
            Evaluate.keypoints_plot,
        ),
        overwrite=overwrite,
    )

    # proj.export_feather("7_scored_behavs", "./scored_csv")
