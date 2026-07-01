"""Make behavysis project."""

from pathlib import Path

from behavysis.constants import DEFAULT_CONFIG_FP, STAGES
from behavysis.models import get_default_config
from behavysis.utils.template_utils import confirm, save_template


def main() -> None:
    """Make behavysis project."""
    if not confirm("Create behavysis project in current directory?"):
        return

    overwrite = confirm("Overwrite existing files?")

    # Make pipeline notebook simple
    if overwrite or not Path("run_pipeline_notebook_simple.py").exists():
        save_template(
            template_name="run_pipeline_notebook_simple.py",
            dst=Path.cwd() / "run_pipeline_notebook_simple.py",
        )
    # Make pipeline script
    if overwrite or not Path("run_pipeline_script.py").exists():
        save_template(
            template_name="run_pipeline_script.py",
            dst=Path.cwd() / "run_pipeline_script.py",
            project_fp_repr=repr(str(Path("project"))),
            config_fp_repr=repr(str(Path("default_config.json"))),
            nprocs=5,
            overwrite=False,
            update_config=True,
            format_vid=True,
            run_dlc=False,
            calculate_parameters=True,
            preprocess=True,
            analyse=True,
            extract_features=True,
            classify_behaviour=True,
            analyse_behaviour=False,
            combine_analysis=False,
            calc_funcs=[
                "start_frame_from_likelihood",
                "stop_frame_from_dur",
                "dur_frames_from_likelihood",
                "px_per_mm",
            ],
            prep_funcs=["interpolate", "start_stop_trim"],
            anal_funcs=["speed", "in_roi", "distance"],
            func_imports={
                "start_frame_from_likelihood",
                "stop_frame_from_dur",
                "dur_frames_from_likelihood",
                "px_per_mm",
                "interpolate",
                "start_stop_trim",
                "speed",
                "in_roi",
                "distance",
            },
        )
    # Make default config
    if overwrite or not Path(DEFAULT_CONFIG_FP).exists():
        Path(DEFAULT_CONFIG_FP).write_text(
            get_default_config().model_dump_json(indent=2),
        )
    # Make folders
    for folder in STAGES:
        Path(folder).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
