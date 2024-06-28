import streamlit as st

from behavysis_pipeline.processes.extract_features import run_simba_subproc
from behavysis_pipeline.processes.run_dlc import run_dlc_subproc

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
