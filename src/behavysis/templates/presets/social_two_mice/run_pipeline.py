import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    from pathlib import Path

    import marimo as mo

    from behavysis import Project
    from behavysis.funcs import (
        analyse_behaviour,
        distance,
        dur_frames_from_likelihood,
        in_roi,
        interpolate,
        px_per_mm,
        start_frame_from_likelihood,
        start_stop_trim,
        stop_frame_from_dur,
        social_distance,
    )
    from behavysis.models import ExperimentConfig
    from behavysis.utils import configure_logger

    configure_logger()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Behavysis — Social Interaction (Two Mice)

    **First Steps**

    * Change `config_fp` below to either
    "default_config_linux.yaml" or "default_config_windows.yaml"
    depending on the machine.
    * Add you videos to the `1_raw_videos` folder.
    """)
    return


@app.cell
def _():
    overwrite = False
    proj_dir = Path.cwd()
    names_ls = [i.stem for i in (proj_dir / "1_raw_videos").iterdir()]
    config_fp = proj_dir / "default_config.yaml"
    nprocs = 4

    mo.accordion(
        {
            "Videos": names_ls,
            "Config": ExperimentConfig.read_yaml(config_fp).model_dump(),
        }
    )
    return config_fp, names_ls, nprocs, overwrite, proj_dir


@app.cell
def _(names_ls, nprocs, proj_dir):
    proj = Project(proj_dir)
    proj.nprocs = nprocs
    proj.import_experiments(names_ls)
    return (proj,)


@app.cell
def _(config_fp, proj):
    proj.update_config(default_config_fp=config_fp)
    return


@app.cell
def _(overwrite, proj):
    proj.format_video(overwrite=overwrite)
    return


@app.cell
def _(proj):
    proj.get_video_metadata()
    return


@app.cell
def _(overwrite, proj):
    proj.run_dlc(gputouse=None, overwrite=overwrite)
    return


@app.cell
def _(proj):
    proj.calculate_parameters(
        funcs=(
            start_frame_from_likelihood,
            stop_frame_from_dur,
            dur_frames_from_likelihood,
            px_per_mm,
        ),
    )
    return


@app.cell
def _(overwrite, proj):
    proj.preprocess(
        funcs=(start_stop_trim, interpolate),
        overwrite=overwrite,
    )
    return


@app.cell
def _(overwrite, proj):
    proj.extract_features(
        overwrite=overwrite,
    )
    return


@app.cell
def _(overwrite, proj):
    proj.classify_behaviour(
        overwrite=overwrite,
    )
    return


@app.cell
def _(overwrite, proj):
    proj.export_behaviour(
        overwrite=overwrite,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Manually check behaviour labels

    Run `behavysis-viewer-app` to verify and correct automated classifications
    before proceeding to behaviour analysis.
    """)
    return


@app.cell
def _(proj):
    proj.analyse(
        funcs=(analyse_behaviour, distance, in_roi, social_distance),
    )
    return


@app.cell
def _(proj):
    proj.combine_analysis()
    proj.collate_analysis()
    return


if __name__ == "__main__":
    app.run()
