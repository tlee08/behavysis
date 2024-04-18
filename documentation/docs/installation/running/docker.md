# Running ba_pipeline with Docker

**Step 1:**
Open the Docker Desktop application. This can be found in the applications folder of the computer once Docker has been downloaded.

<p align="center">
    <img src="../../figures/docker_icon.png" alt="docker_icon" title="docker_icon" style="width:10%">
</p>

It may take Docker around approx 1-2 minutes to finish opening. Once Docker Desktop is open, a screen like this should be visible:

<p align="center">
    <img src="../../figures/docker_window.png" alt="docker_window" title="docker_window" style="width:70%">
</p>

**Step 2:**
Open a terminal or command line, which should look like one of the windows in the figure below.

<p align="center">
    <img src="../../figures/terminals.png" alt="terminals" title="terminals" style="width:80%">
</p>

**Step 3:**
Navigate to the folder where you have (or would like to store) your Jupyter Notebook to analyse experiments with.
This can be done with the following command:

```zsh
cd /path/to/my_notebooks_folder
```

**Step 4:**
An example of using this command to navigate to the "example" folder that was found in step 4 is shown below.

<p align="center">
    <img src="../../figures/terminal_cd.png" alt="terminal_cd" title="terminal_cd" style="width:50%">
</p>

**Step 5:**
To run the `ba_pipeline` program, enter the following command into the terminal:

```zsh
docker run --name ba_pipeline_container -it --rm -p 8888:8888 -v "${PWD}":/app ba_pipeline:core
```

The terminal should output something similar to the figure below.

<p align="center">
    <img src="../../figures/terminal_run.png" alt="terminal_run" title="terminal_run" style="width:40%">
</p>

**Step 6:**
To access the Jupyter-Lab server to interact with the `ba_pipeline` program, open a browser (e.g., Chrome, Safari, FireFox, Brave, Opera) and enter the following URL:

```zsh
http://127.0.0.1:8888/lab
```

An example of how the Jupyter Notebook should look like is shown in the figure below.

<p align="center">
    <img src="../../figures/terminal_jupyter.png" alt="terminal_jupyter" title="terminal_jupyter" style="width:50%">
</p>

**Step 7:**
You can now use the `ba_pipeline` program through a Jupyter Notebook. For how to use this, please see the [tutorials](../../tutorials/usage_examples/one_project.md).
