# Running behavysis_pipeline

**Step 1:**

**_FOR WINDOWS_** - Open an Anaconda PowerShell Prompt window (make sure this is specifically this application - some other command prompt windows make look similar but they won't work). An image of this application is shown below.

<p align="center">
    <img src="../../figures/windows_conda_powershell.png" alt="windows_conda_powershell" title="windows_conda_powershell" style="width:40%">
</p>

**_FOR MAC OR LINUX_** - Open a terminal window. An image of this application is shown below.

<p align="center">
    <img src="../../figures/mac_terminal.png" alt="mac_terminal" title="mac_terminal" style="width:40%">
</p>

**Step 2:**
Activate the program environment with the following command:

```zsh
conda activate ba_env
```

**Step 3:**
Navigate to the folder where you have (or would like to store) your Jupyter Notebook to analyse experiments with.
This can be done with the following command:

```zsh
cd /path/to/my_notebooks_folder
```

**Step 4:**
Open a Jupyter-Lab server with the following command.

```zsh
jupyter-lab
```

**Step 5:**
To access the Jupyter-Lab server to interact with the `behavysis_pipeline` program, open a browser (e.g., Chrome, Safari, FireFox, Brave, Opera) and enter the following URL:

```zsh
http://127.0.0.1:8888/lab
```

An example of how the Jupyter Notebook should look like is shown in the figure below.

<p align="center">
    <img src="../../figures/terminal_jupyter.png" alt="terminal_jupyter" title="terminal_jupyter" style="width:80%">
</p>

**Step 6:**
Please ensure that it the "ba_env" kernel is used.
To change the notebook's kernel to "ba_env", select "Kernel" > "Change Kernel..." in the top left menu bar. Then select the "ba_env" from the prompt.

<p align="center">
    <img src="../../figures/jupyter_kernel1.png" alt="terminal_jupyter" title="terminal_jupyter" style="width:70%">
</p>
<p align="center">
    <img src="../../figures/jupyter_kernel2.png" alt="terminal_jupyter" title="terminal_jupyter" style="width:50%">
</p>

**Step 7:**
You can now use the `behavysis_pipeline` program through a Jupyter Notebook. For how to use this, please see the [tutorials](../../tutorials/usage_examples/one_project.md).
