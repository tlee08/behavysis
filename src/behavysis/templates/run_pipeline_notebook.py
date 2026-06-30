import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    from collections.abc import Callable
    from pathlib import Path

    import marimo as mo

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
    from behavysis.models import ExperimentConfig, get_default_config
    from behavysis.utils.template_utils import render_template


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Behavysis Pipeline Runner
    """)
    return


@app.function
def create_funcs_checkbox_list(
    funcs_ls: list[tuple[Callable, bool]],
) -> list[tuple[Callable, mo.ui.checkbox]]:
    funcs_checkbox_dict = [
        (
            _func,
            mo.ui.checkbox(label=str(_func.__name__), value=_is_run),
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
    return


@app.cell
def _():
    overwrite = mo.ui.switch(label="Overwrite files")
    project_fp = mo.ui.text(
        label="Project folder", value=str(Path.cwd()), full_width=True
    )
    config_fp = mo.ui.text(
        label="Default config",
        value=str(Path.cwd() / DEFAULT_CONFIG_FP),
        full_width=True,
    )
    nprocs = mo.ui.number(label="Number of parallel processes", value=5)
    return config_fp, nprocs, overwrite, project_fp


@app.cell
def _(config_fp, nprocs, overwrite, project_fp):
    mo.vstack(
        [
            overwrite,
            project_fp,
            config_fp,
            nprocs,
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Inspect default config
    """)
    return


@app.cell
def _(config_fp):
    config_fp_path = Path(config_fp.value)

    mo.stop(not config_fp_path.exists(), mo.md("Config file does not exist!"))

    ExperimentConfig.model_validate_json(config_fp_path.read_text())
    return


@app.cell
def _():
    mo.accordion(
        {
            "See default configs template": get_default_config().model_dump(),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Choose functions to run
    """)
    return


@app.cell
def _():
    # Each Step
    update_config_checkbox = mo.ui.checkbox(label="Step 0: Update config", value=True)
    format_vid_checkbox = mo.ui.checkbox(
        label="Step 1: Format videos",
        value=True,
    )
    run_dlc_checkbox = mo.ui.checkbox(
        label="Step 2: Run DeepLabCut pose estimation",
        value=True,
    )
    calculate_parameters_checkbox = mo.ui.checkbox(
        label="Step 3: Calculate experiment parameters",
        value=True,
    )
    preprocess_checkbox = mo.ui.checkbox(
        label="Step 4: Preprocess keypoints",
        value=True,
    )
    analyse_checkbox = mo.ui.checkbox(
        label="Step 5: Simple Analysis",
        value=True,
    )
    extract_features_checkbox = mo.ui.checkbox(
        label="Step 6: Extract features for classifier",
        value=True,
    )
    classify_behaviour_checkbox = mo.ui.checkbox(
        label="Step 7: Classify behaviours",
        value=True,
    )
    manually_check_labels_msg = mo.md(
        "Step 8: Run `behavysis-viewer` to verify the classified behaviour",
    )
    analyse_behaviour_checkbox = mo.ui.checkbox(
        label="Step 9: Make the analysis for the verified behaviour",
        value=True,
    )
    combine_analysis_checkbox = mo.ui.checkbox(
        label="Step 10: Analyze results",
        value=True,
    )

    # For choosing funcs
    calculate_parameters_funcs_ls = create_funcs_checkbox_list(
        [
            (start_frame_from_likelihood, True),
            (stop_frame_from_dur, True),
            (dur_frames_from_likelihood, True),
            (px_per_mm, True),
        ]
    )
    preprocess_funcs_ls = create_funcs_checkbox_list(
        [
            (start_stop_trim, True),
            (interpolate, True),
        ]
    )
    analyse_funcs_ls = create_funcs_checkbox_list(
        [
            (in_roi, True),
            (speed, True),
            (distance, True),
        ]
    )

    # Run button
    run_btn = mo.ui.run_button(label="Run Pipeline")
    return (
        analyse_behaviour_checkbox,
        analyse_checkbox,
        analyse_funcs_ls,
        calculate_parameters_checkbox,
        calculate_parameters_funcs_ls,
        classify_behaviour_checkbox,
        combine_analysis_checkbox,
        extract_features_checkbox,
        format_vid_checkbox,
        manually_check_labels_msg,
        preprocess_checkbox,
        preprocess_funcs_ls,
        run_btn,
        run_dlc_checkbox,
        update_config_checkbox,
    )


@app.cell
def _(
    analyse_behaviour_checkbox,
    analyse_checkbox,
    analyse_funcs_ls,
    calculate_parameters_checkbox,
    calculate_parameters_funcs_ls,
    classify_behaviour_checkbox,
    combine_analysis_checkbox,
    extract_features_checkbox,
    format_vid_checkbox,
    manually_check_labels_msg,
    preprocess_checkbox,
    preprocess_funcs_ls,
    run_btn,
    run_dlc_checkbox,
    update_config_checkbox,
):
    mo.vstack(
        [
            update_config_checkbox,
            format_vid_checkbox,
            run_dlc_checkbox,
            calculate_parameters_checkbox,
            mo.callout(get_checkbox_list(calculate_parameters_funcs_ls), kind="info"),
            preprocess_checkbox,
            mo.callout(get_checkbox_list(preprocess_funcs_ls), kind="info"),
            analyse_checkbox,
            mo.callout(get_checkbox_list(analyse_funcs_ls), kind="info"),
            extract_features_checkbox,
            classify_behaviour_checkbox,
            manually_check_labels_msg,
            analyse_behaviour_checkbox,
            combine_analysis_checkbox,
            run_btn,
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Running Project
    """)
    return


@app.cell
def _(
    analyse_behaviour_checkbox,
    analyse_checkbox,
    analyse_funcs_ls,
    calculate_parameters_checkbox,
    calculate_parameters_funcs_ls,
    classify_behaviour_checkbox,
    combine_analysis_checkbox,
    config_fp,
    extract_features_checkbox,
    format_vid_checkbox,
    nprocs,
    overwrite,
    preprocess_checkbox,
    preprocess_funcs_ls,
    project_fp,
    run_btn,
    run_dlc_checkbox,
    update_config_checkbox,
):
    mo.stop(
        not run_btn.value,
        mo.md("""Click the 'Run Pipeline' button once you're happy to run"""),
    )

    names_ls = [i.name for i in (Path(project_fp.value) / "1_raw_videos").iterdir()]
    proj = Project(Path(project_fp.value))
    proj.nprocs = nprocs.value
    proj.import_experiments(names_ls)

    if update_config_checkbox.value:
        proj.update_config(
            default_config_fp=config_fp.value,
            overwrite="user",
        )

    if format_vid_checkbox.value:
        proj.format_video(
            overwrite=overwrite.value,
        )

    if run_dlc_checkbox.value:
        proj.run_dlc(
            gputouse=None,
            overwrite=overwrite.value,
        )

    if calculate_parameters_checkbox.value:
        proj.calculate_parameters(
            funcs=get_funcs_to_run_list(calculate_parameters_funcs_ls),
        )

    if preprocess_checkbox.value:
        proj.preprocess(
            funcs=get_funcs_to_run_list(preprocess_funcs_ls),
            overwrite=overwrite.value,
        )

    if analyse_checkbox.value:
        proj.analyse(
            funcs=get_funcs_to_run_list(analyse_funcs_ls),
        )
        proj.combine_analysis()
        proj.collate_analysis()

    if extract_features_checkbox.value:
        proj.extract_features(
            overwrite=overwrite.value,
        )

    if classify_behaviour_checkbox.value:
        proj.classify_behaviour(overwrite=overwrite.value)
        proj.export_behaviour(overwrite=overwrite.value)

    # manually_check_labels_msg

    if analyse_behaviour_checkbox.value:
        proj.analyse_behaviour()

    if combine_analysis_checkbox.value:
        proj.combine_analysis()
        proj.collate_analysis()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Export standalone pipeline script

    Generates a standalone Python script that reproduces the configured pipeline.
    """)
    return


@app.cell
def _(
    analyse_behaviour_checkbox,
    analyse_checkbox,
    analyse_funcs_ls,
    calculate_parameters_checkbox,
    calculate_parameters_funcs_ls,
    classify_behaviour_checkbox,
    combine_analysis_checkbox,
    config_fp,
    extract_features_checkbox,
    format_vid_checkbox,
    nprocs,
    overwrite,
    preprocess_checkbox,
    preprocess_funcs_ls,
    project_fp,
    run_dlc_checkbox,
    update_config_checkbox,
):
    def _build_script():
        calc_funcs = get_funcs_to_run_list(calculate_parameters_funcs_ls)
        prep_funcs = get_funcs_to_run_list(preprocess_funcs_ls)
        anal_funcs = get_funcs_to_run_list(analyse_funcs_ls)

        all_func_names = {f.__name__ for f in calc_funcs + prep_funcs + anal_funcs}

        return render_template(
            "run_pipeline_script.py",
            project_fp_repr=repr(str(project_fp.value)),
            config_fp_repr=repr(str(config_fp.value)),
            nprocs=nprocs.value,
            overwrite=overwrite.value,
            update_config=update_config_checkbox.value,
            format_vid=format_vid_checkbox.value,
            run_dlc=run_dlc_checkbox.value,
            calculate_parameters=calculate_parameters_checkbox.value
            and bool(calc_funcs),
            preprocess=preprocess_checkbox.value and bool(prep_funcs),
            analyse=analyse_checkbox.value and bool(anal_funcs),
            extract_features=extract_features_checkbox.value,
            classify_behaviour=classify_behaviour_checkbox.value,
            analyse_behaviour=analyse_behaviour_checkbox.value,
            combine_analysis=combine_analysis_checkbox.value,
            calc_funcs=[f.__name__ for f in calc_funcs],
            prep_funcs=[f.__name__ for f in prep_funcs],
            anal_funcs=[f.__name__ for f in anal_funcs],
            func_imports=all_func_names,
        )

    export_download = mo.download(
        data=lambda: _build_script().encode("utf-8"),
        filename="run_pipeline.py",
        mimetype="text/x-python",
        label="Export Pipeline Code (.py)",
    )

    mo.hstack(
        [
            mo.md("Click to download the configured pipeline as a standalone script:"),
            export_download,
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
