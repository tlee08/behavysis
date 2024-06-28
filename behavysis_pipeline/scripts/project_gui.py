import json
import os
import subprocess

import pandas as pd
import streamlit as st
from behavysis_core.data_models.experiment_configs import ExperimentConfigs
from behavysis_core.mixins.io_mixin import IOMixin

from behavysis_pipeline.pipeline.project import Project
from behavysis_pipeline.processes.calculate_params import CalculateParams
from behavysis_pipeline.processes.extract_features import run_simba_subproc
from behavysis_pipeline.processes.preprocess import Preprocess
from behavysis_pipeline.processes.run_dlc import run_dlc_subproc

#####################################################################
# Pipeline Functions (callbacks)
#####################################################################


def status_wrapper(func):
    def wrapper(*args, **kwargs):
        with st.status("Processing...", expanded=True) as status:
            func(*args, **kwargs)
            status.update(label="Done!", state="complete", expanded=True)
            # Attempting to read the corresponding diagnostics df
            try:
                proj = st.session_state.get("proj", None)
                if proj is not None:
                    df = proj.load_diagnostics(func.__name__)
                    st.dataframe(df)
            except FileNotFoundError:
                pass

    return wrapper


def get_class_methods(cls):
    return [
        getattr(cls, method)
        for method in dir(cls)
        if callable(getattr(cls, method)) and not method.startswith("__")
    ]


def init_project(proj_dir: str):
    st.session_state["proj"] = Project(proj_dir)
    st.success("Project Initialised")


@status_wrapper
def import_experiments(proj: Project):
    proj.import_experiments()
    st.success("Experiments imported")
    st.success(f"Experiments: \n\n{"\n".join(proj.experiments)}")


def upload_configs(configs_f):
    if configs_f is not None:
        configs = json.loads(configs_f.read().decode("utf-8"))
        st.success("Config file uploaded")
        st.write("Configs:")
        st.json(configs, expanded=False)
        st.session_state["configs"] = configs


@status_wrapper
def update_configs(proj: Project, configs: dict, overwrite: str):
    # Writing configs to temp file
    configs_fp = os.path.join(proj.root_dir, ".temp", "temp_configs.json")
    configs_model = ExperimentConfigs.model_validate(configs)
    configs_model.write_json(configs_fp)
    # Updatng configs
    proj.update_configs(configs_fp, overwrite)
    # Removing temp file
    IOMixin.silent_rm(configs_fp)
    # Success message
    st.success("Configs Updated")


@status_wrapper
def calculate_params(proj: Project, method_checks: dict):
    # Getting list of methods to run
    methods = [method for method, check in method_checks.values() if check]
    # Running methods
    proj.calculate_params(methods)
    # Success message
    st.success("Parameters Calculated")


@status_wrapper
def preprocess(proj: Project, method_checks: dict, overwrite: bool):
    # Getting list of methods to run
    methods = [method for method, check in method_checks.values() if check]
    # Running methods
    proj.preprocess(methods, overwrite)
    # Success message
    st.success("Parameters Calculated")


#####################################################################
# Streamlit pages
#####################################################################


def page_header(title):
    # Recalling session state variables
    proj: Project = st.session_state.get("proj", None)
    root_dir = proj.root_dir if proj is not None else "UNSET"
    # Title
    st.title(title)
    # Project overview
    with st.sidebar:
        st.subheader(f"project: {root_dir}")
        if proj is not None:
            st.dataframe(
                pd.DataFrame(
                    proj.experiments.keys(),
                    columns=["Experiments"],
                )
            )
        else:
            st.write("No project initialised")


def page_init_project():
    page_header("Init Project")
    # Recalling session state variables
    proj: Project = st.session_state.get("proj", None)
    root_dir = proj.root_dir if proj is not None else "UNSET"
    # Page description
    st.write("This page is for making a new Behavysis project.")
    # Text input: project root folder
    proj_dir = st.text_input("Root Directory", ".")
    proj_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/TimLee/resources/project_ma"
    # Button: Get project
    btn_proj = st.button(
        "Init Project",
        on_click=init_project,
        args=(proj_dir,),
    )
    # Button: Import experiments
    btn_import = st.button(
        "Import Experiments",
        on_click=import_experiments,
        args=(proj,),
        disabled=proj is None,
    )


def page_update_configs():
    page_header("Update Configs")
    # Recalling session state variables
    proj: Project = st.session_state.get("proj", None)
    configs = st.session_state.get("configs", None)
    # Page description
    st.write("This page is for making a new Behavysis project.")
    # User input: selecting default configs file
    configs_f = st.file_uploader(
        "Upload Default Configs",
        type=["json"],
        disabled=proj is None,
    )
    upload_configs(configs_f)
    # Select box: overwrite option
    overwrite_selected = st.selectbox(
        "Select an option",
        options=["user", "all"],
        disabled=configs is None,
    )
    # Button: Update configs
    st.button(
        "Update Configs",
        on_click=update_configs,
        args=(proj, configs, overwrite_selected),
        disabled=configs is None,
    )


def page_run_dlc():
    page_header("Run DLC")
    # TODO: have a selector for DLC model
    # Recalling session state variables
    proj: Project = st.session_state.get("proj", None)
    # Page description
    st.write("This page is for making a new Behavysis project.")


def page_calculate_params():
    page_header("Calculate Params")
    # TODO: For each function, have a configs updater.
    # Recalling session state variables
    proj: Project = st.session_state.get("proj", None)
    # Page description
    st.write(
        "Calculate the project's inherent parameters "
        + "from the video and DLC keypoints data"
    )
    # List of checkboxes for each method
    st.subheader("Select Methods to Run")
    methods = get_class_methods(CalculateParams)
    method_checks = {
        method.__name__: (
            method,
            st.checkbox(method.__name__, disabled=proj is None),
        )
        for method in methods
    }
    # Button: Run selected methods
    st.button(
        "Calculate Parameters",
        on_click=calculate_params,
        args=(proj, method_checks),
        disabled=proj is None,
    )


def page_preprocess():
    page_header("Preprocess")
    # TODO: For each function, have a configs updater.
    # Recalling session state variables
    proj: Project = st.session_state.get("proj", None)
    # Page description
    st.write(
        "Calculate the project's inherent parameters "
        + "from the video and DLC keypoints data"
    )
    # List of checkboxes for each method
    st.subheader("Select Methods to Run")
    methods = get_class_methods(Preprocess)
    method_checks = {
        method.__name__: (
            method,
            st.checkbox(method.__name__, disabled=proj is None),
        )
        for method in methods
    }
    st.subheader("Overwrite Existing Files")
    # Checkbox: Overwrite
    overwrite = st.checkbox(
        "Overwrite",
        disabled=proj is None,
    )
    # Button: Run selected methods
    st.button(
        "Calculate Parameters",
        on_click=preprocess,
        args=(proj, method_checks, overwrite),
        disabled=proj is None,
    )


def page_extract_features():
    page_header("Extract Features")
    # Recalling session state variables
    proj: Project = st.session_state.get("proj", None)
    # Page description
    st.write("This page is for making a new Behavysis project.")


def page_classify_behaviours():
    page_header("Classify Behaviours")
    # TODO: have a selector for behaviour classifier
    # Recalling session state variables
    proj: Project = st.session_state.get("proj", None)
    # Page description
    st.write("This page is for making a new Behavysis project.")


#####################################################################
# Streamlit application
#####################################################################


def main():
    st.title("Behavysis Pipeline Runner")

    pg = st.navigation(
        [
            st.Page(page_init_project, title="init_project"),
            st.Page(page_update_configs, title="update_configs"),
            st.Page(page_run_dlc, title="run_dlc"),
            st.Page(page_calculate_params, title="calculate_params"),
            st.Page(page_preprocess, title="preprocess"),
            st.Page(page_extract_features, title="extract_features"),
            st.Page(page_classify_behaviours, title="classify_behaviours"),
        ]
    )

    pg.run()


def main_old():
    # Title of the application
    st.title("Behavysis Pipeline Runner")

    # Sidebar for navigation
    app_mode = st.sidebar.selectbox(
        "Choose the application mode", ["Run DLC Subprocess", "Run SimBA Subprocess"]
    )

    if app_mode == "Run DLC Subprocess":
        st.header("Run DLC Subprocess")
        # Input fields for the user to fill in the required parameters
        model_fp = st.text_input("Model File Path", "")
        in_fp_ls = st.text_area("Input File Paths (comma-separated)", "").split(",")
        dlc_out_dir = st.text_input("DLC Output Directory", "")
        temp_dir = st.text_input("Temporary Directory", "")
        gputouse = st.number_input("GPU to Use", min_value=0, value=0, step=1)

        if st.button("Run DLC Subprocess"):
            # Assuming run_dlc_subproc function handles execution and error logging
            run_dlc_subproc(model_fp, in_fp_ls, dlc_out_dir, temp_dir, gputouse)
            st.success("DLC Subprocess Completed Successfully")

    elif app_mode == "Run SimBA Subprocess":
        st.header("Run SimBA Subprocess")
        # Input fields for SimBA subprocess
        simba_dir = st.text_input("SimBA Directory", "")
        dlc_dir = st.text_input("DLC Directory", "")
        configs_dir = st.text_input("Configs Directory", "")
        temp_dir = st.text_input("Temporary Directory for SimBA", "")
        cpid = st.number_input("Custom Process ID", min_value=0, value=0, step=1)

        if st.button("Run SimBA Subprocess"):
            # Assuming run_simba_subproc function handles execution
            message = run_simba_subproc(simba_dir, dlc_dir, configs_dir, temp_dir, cpid)
            st.success(message)


def run_script():
    curr_fp = os.path.abspath(__file__)
    subprocess.run(["streamlit", "run", curr_fp])


if __name__ == "__main__":
    main()
