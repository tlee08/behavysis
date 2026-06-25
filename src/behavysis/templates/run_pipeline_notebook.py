import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    from collections.abc import Callable
    from pathlib import Path

    import marimo as mo

    from behavysis import Project
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Behavysis Pipeline Runner
    """)


@app.function
def create_funcs_checkbox_list(
    funcs_ls: list[tuple[Callable, bool]],
) -> list[Callable, object]:
    funcs_checkbox_dict = [
        (
            _func,
            mo.ui.checkbox(label=_func.__name__, value=_is_run),
        )
        for _func, _is_run in funcs_ls
    ]
    return funcs_checkbox_dict


@app.function
def get_checkbox_list(funcs_checkbox_ls: list[tuple[object, Callable]]):
    return mo.vstack(
        [
            mo.hstack([_checkbox, _func.__doc__.split("\n")[0]])
            for _func, _checkbox in funcs_checkbox_ls
        ]
    )


@app.function
def get_funcs_to_run_list(
    funcs_checkbox_ls: list[tuple[object, Callable]],
):
    return [_func for _func, _checkbox in funcs_checkbox_ls if _checkbox.value]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Set up pipeline to run
    """)


@app.cell
def _(DEFAULT_CONFIG_FP):
    overwrite = mo.ui.switch(label="Overwrite files")
    project_fp = mo.ui.text(label="Project folder", value=Path.cwd(), full_width=True)
    configs_fp = mo.ui.text(
        label="Project folder",
        value=Path.cwd() / DEFAULT_CONFIG_FP,
        full_width=True,
    )
    nprocs = mo.ui.number(label="Number of parallel processes", value=5)

    run_btn = mo.ui.run_button(label="Run Pipeline")
    return configs_fp, nprocs, overwrite, project_fp, run_btn


@app.cell
def _(configs_fp, nprocs, overwrite, project_fp, run_btn):
    mo.vstack(
        [
            overwrite,
            project_fp,
            configs_fp,
            nprocs,
            run_btn,
        ]
    )


@app.cell
def _():
    update_config_checkbox = mo.ui.checkbox(label="Step 0: Update configs")
    format_vid_checkbox = mo.ui.checkbox(label="Step 1: Format videos")
    format_vid_checkbox = mo.ui.checkbox(label="Step 2: Format videos")
    calculate_experiment_checkbox = mo.ui.checkbox(
        label="Step 3: Calculate experiment parameters"
    )
    preprocess_checkbox = mo.ui.checkbox(label="Step 4: Preprocess keypoints")
    analysis_checkbox = mo.ui.checkbox(label="Step 5: Simple Analysis")
    extract_features_checkbox = mo.ui.checkbox(
        label="Step 6: Extract features for classifier"
    )
    classify_behaviour = mo.ui.checkbox(label="Step 7: Classify behaviours")
    manually_check_labels = mo.md(
        "Step 8: Run `behavysis-viewer` to verify the classified behaviour"
    )
    analyse_behaviour = mo.ui.checkbox(
        label="Step 9: Make the analysis for the verified behaviour"
    )
    mo.ui.checkbox(label="Step 10: Analyze results")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load Project
    """)


@app.cell
def _(nprocs, project_fp, run_btn):
    if run_btn.value:
        proj = Project(Path.cwd(project_fp.value))
        proj.nprocs = nprocs.value
        proj.import_experiments()
    return (proj,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 0: Update configs
    """)


@app.cell
def _(configs_fp, proj):
    proj.update_config(
        default_config_fp=configs_fp.value,
        overwrite="user",
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Format videos
    """)


@app.cell
def _(overwrite, proj):
    proj.format_vid(
        overwrite=overwrite.value,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Run DeepLabCut pose estimation
    """)


@app.cell
def _(overwrite, proj):
    proj.run_dlc(
        gputouse=None,
        overwrite=overwrite.value,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 3: Calculate experiment parameters
    """)


@app.cell
def _():
    calculate_parameters_funcs_ls = create_funcs_checkbox_list(
        [
            (start_frame_from_likelihood, True),
            (stop_frame_from_dur, True),
            (dur_frames_from_likelihood, True),
            (px_per_mm, True),
        ]
    )

    get_checkbox_list(calculate_parameters_funcs_ls)
    return (calculate_parameters_funcs_ls,)


@app.cell
def _(calculate_parameters_funcs_ls, proj):
    proj.calculate_parameters(
        funcs=get_funcs_to_run_list(calculate_parameters_funcs_ls),
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 4: Preprocess keypoints
    """)


@app.cell
def _():
    preprocess_funcs_ls = create_funcs_checkbox_list(
        [
            (start_stop_trim, True),
            (interpolate, True),
        ]
    )

    get_checkbox_list(preprocess_funcs_ls)
    return (preprocess_funcs_ls,)


@app.cell
def _(overwrite, preprocess_funcs_ls, proj):
    proj.preprocess(
        funcs=get_funcs_to_run_list(preprocess_funcs_ls),
        overwrite=overwrite.value,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 5: Simple Analysis
    """)


@app.cell
def _():
    analyse_funcs_ls = create_funcs_checkbox_list(
        [
            (in_roi, True),
            (speed, True),
            (distance, True),
        ]
    )

    get_checkbox_list(analyse_funcs_ls)
    return (analyse_funcs_ls,)


@app.cell
def _(analyse_funcs_ls, proj):
    proj.analyse(
        funcs=get_funcs_to_run_list(analyse_funcs_ls),
    )


@app.cell
def _(proj):
    proj.combine_analysis()
    proj.collate_analysis()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 6: Extract features for classifier

    Only run step 6-9 if you are using classified behaviour pipeline
    """)


@app.cell
def _(overwrite, proj):
    proj.extract_features(
        overwrite=overwrite.value,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 7: Classify behaviours
    """)


@app.cell
def _(overwrite, proj):
    # Requires: user.classify_behaviour with trained model paths
    proj.classify_behaviour(overwrite=overwrite.value)
    proj.export_behaviour(overwrite=overwrite.value)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 8: Run `behavysis-viewer` to verify the classified behaviour
    """)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 9: Make the analysis for the verified behaviour
    """)


@app.cell
def _(proj):
    proj.analyse_behaviour()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 10: Analyze results
    """)


@app.cell
def _(proj):
    proj.combine_analysis()
    proj.collate_analysis()


if __name__ == "__main__":
    app.run()
