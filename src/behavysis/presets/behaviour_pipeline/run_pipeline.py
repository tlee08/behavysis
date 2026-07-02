import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    from pathlib import Path

    import marimo as mo

    from behavysis import Project
    from behavysis.funcs import (
        distance,
        dur_frames_from_likelihood,
        freezing,
        in_roi,
        interpolate,
        px_per_mm,
        speed,
        start_frame_from_likelihood,
        start_stop_trim,
        stop_frame_from_dur,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Behavysis — Full Behaviour Pipeline""")


@app.cell
def _():
    overwrite = False
    proj_dir = Path.cwd()
    names_ls = [i.stem for i in (proj_dir / "1_raw_videos").iterdir()]
    config_fp = proj_dir / "default_config.yaml"
    nprocs = 4
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


@app.cell
def _(overwrite, proj):
    proj.format_video(overwrite=overwrite)


@app.cell
def _(overwrite, proj):
    proj.run_dlc(gputouse=None, overwrite=overwrite)


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


@app.cell
def _(overwrite, proj):
    proj.preprocess(
        funcs=(start_stop_trim, interpolate),
        overwrite=overwrite,
    )


@app.cell
def _(overwrite, proj):
    proj.extract_features(overwrite=overwrite)


@app.cell
def _(overwrite, proj):
    proj.classify_behaviour(overwrite=overwrite)
    proj.export_behaviour(overwrite=overwrite)


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Manually check behaviour labels

Run `behavysis-viewer-app` to verify and correct automated classifications
before proceeding to behaviour analysis.
""")


@app.cell
def _(proj):
    proj.analyse_behaviour()
    proj.combine_analysis()
    proj.collate_analysis()


@app.cell
def _(proj):
    proj.analyse(
        funcs=(speed, distance, freezing, in_roi),
    )
    proj.combine_analysis()
    proj.collate_analysis()


if __name__ == "__main__":
    app.run()
