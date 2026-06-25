"""Behavysis Pipeline Script.

This script runs the full behavioral analysis pipeline.

Documentation: https://tlee08.github.io/behavysis/
Config Reference: https://tlee08.github.io/behavysis/tutorials/config_json/
"""

from pathlib import Path

from behavysis import Project
from behavysis.constants import DEFAULT_CONFIG_FP
from behavysis.funcs import (
    distance,
    dur_frames_from_likelihood,
    in_roi,
    interpolate,
    px_per_mm,
    speed,
    start_frame_from_likelihood,
    start_stop_trim,
    stop_frame_from_dur,
)

if __name__ == "__main__":
    overwrite = False

    proj_dir = Path.cwd()
    proj = Project(proj_dir)
    proj.import_experiments()
    proj.nprocs = 5

    # Apply config settings to all experiments
    # See: https://tlee08.github.io/behavysis/tutorials/config_json/
    default_config_fp = proj_dir / DEFAULT_CONFIG_FP
    proj.update_config(
        default_config_fp=default_config_fp,
        overwrite="user",
    )

    # Step 1: Format videos
    proj.format_video(overwrite=overwrite)

    # Step 2: Run DeepLabCut pose estimation
    proj.run_dlc(
        gputouse=None,
        overwrite=overwrite,
    )

    # Step 3: Calculate experiment parameters
    proj.calculate_parameters(
        funcs=(
            # Auto-detect start frame from keypoint likelihood
            # Requires: user.calculate_params.from_likelihood
            start_frame_from_likelihood,
            # Set stop frame from fixed duration
            # Requires: user.calculate_params.stop_frame_from_dur.dur_sec
            stop_frame_from_dur,
            # Calculate total frames
            dur_frames_from_likelihood,
            # Calculate pixels per millimeter
            # Requires: user.calculate_params.px_per_mm
            px_per_mm,
        )
    )

    # Step 4: Preprocess keypoints
    proj.preprocess(
        funcs=(
            # Trim to start/stop frames
            start_stop_trim,
            # Interpolate low-confidence keypoints
            # Requires: user.preprocess.interpolate.pcutoff
            interpolate,
        ),
        overwrite=overwrite,
    )

    proj.analyse(
        funcs=(
            in_roi,
            speed,
            distance,
        )
    )

    # Step 5: Extract features for classifier
    # Only run step 5-8 if you are using classified behavs pipeline
    proj.extract_features(overwrite=overwrite)

    # Step 6: Classify behaviors
    # Requires: user.classify_behavs with trained model paths
    proj.classify_behavs(overwrite=overwrite)
    proj.export_behavs(overwrite=overwrite)

    # Step 7: Run `behavysis-viewer` to verify the classified behavs

    # Step 8: Make the analysis for the verified behavs
    proj.analyse_behavs()

    # Step 9: Analyze results
    proj.combine_analysis()
    proj.collate_analysis()

    # Step 10: Generate evaluation video
    # Optional
    proj.evaluate_vid(overwrite=overwrite)

    print("Pipeline complete!")
