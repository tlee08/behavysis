"""_summary_"""

import os

from behavysis.pipeline.project import Project
from behavysis.processes.analyse import Analyse
from behavysis.processes.calculate_params import CalculateParams
from behavysis.processes.format_vid import FormatVid
from behavysis.processes.preprocess import Preprocess

if __name__ == "__main__":
    overwrite = True

    proj_dir = os.path.join(".")
    proj = Project(proj_dir)
    proj.import_experiments()
    exp = proj.experiments[1]

    proj.nprocs = 5

    default_configs_fp = os.path.join(proj_dir, "default_configs.json")
    proj.update_configs(
        default_configs_fp,
        overwrite="user",
    )

    proj.format_vid(overwrite=overwrite)

    proj.format_vid_metadata()

    proj.run_dlc(
        gputouse=None,
        overwrite=overwrite,
    )

    proj.calculate_parameters(
        (
            CalculateParams.dlc_scorer_name,
            CalculateParams.start_frame_from_likelihood,
            CalculateParams.stop_frame_from_dur,
            CalculateParams.stop_frame_from_likelihood,
            CalculateParams.dur_frames_from_likelihood,
            CalculateParams.px_per_mm,
        )
    )

    proj.collate_auto_configs()

    proj.preprocess(
        (
            Preprocess.start_stop_trim,
            Preprocess.interpolate,
        ),
        overwrite=overwrite,
    )

    proj.analyse(
        (
            Analyse.in_roi,
            Analyse.speed,
            Analyse.social_distance,
            Analyse.freezing,
        )
    )

    proj.extract_features(overwrite)
    proj.classify_behavs(overwrite)
    proj.export_behavs(overwrite)

    proj.analyse_behavs()

    proj.combine_analysis(overwrite)

    proj.evaluate_vid(overwrite)

    # proj.export2csv("7_scored_behavs", "./scored_csv", overwrite)
