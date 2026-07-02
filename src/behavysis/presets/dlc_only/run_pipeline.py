import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    from pathlib import Path

    import marimo as mo

    from behavysis import Project


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Behavysis — DLC Keypoint Tracking Only""")


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


if __name__ == "__main__":
    app.run()
