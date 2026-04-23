# Running behavysis

This guide explains how to run behavysis after installation.

!!! tip "Prerequisites"
    - behavysis installed ([Installing Guide](installing.md))
    - Project folder set up ([Setup Tutorial](../tutorials/setup.md))

---

## Quick Start

### 1. Activate Environment

```bash
conda activate behavysis
```

### 2. Run Python or Jupyter

=== "Python Script"

    ```bash
    python my_analysis.py
    ```

=== "Jupyter Notebook"

    ```bash
    jupyter lab
    ```
    
    Then open `http://127.0.0.1:8888/lab` in your browser.

=== "Python Interactive"

    ```bash
    python
    >>> from behavysis import Project
    >>> proj = Project("./my_project")
    ```

---

## Basic Workflow

Here's a minimal example to get started:

```python
from behavysis import Project
from behavysis.processes import *

# Initialize project
proj = Project("./my_project")

# Import experiments (scans 1_raw_vid/ folder)
proj.import_experiments()

# Update configurations
proj.update_configs("./default_config.json", overwrite="user")

# Run pipeline
proj.format_vid(overwrite=True)
proj.run_dlc(gputouse=0, overwrite=True)
proj.calculate_parameters((
    CalculateParams.start_frame,
    CalculateParams.stop_frame,
    CalculateParams.px_per_mm,
))
proj.preprocess((
    Preprocess.start_stop_trim,
    Preprocess.interpolate,
), overwrite=True)

# Analyze
proj.analyse((
    Analyse.thigmotaxis,
    Analyse.speed,
))
```

---

## Using Jupyter Notebooks

### Setup

```bash
# Ensure behavysis kernel is available
conda activate behavysis
python -m ipykernel install --user --name behavysis

# Launch Jupyter
jupyter lab
```

### Select Kernel

In JupyterLab:
1. Click the kernel name (top right)
2. Select **"Python (behavysis)"**

### Example Notebook Structure

```python
# Cell 1: Imports
from behavysis import Project
from behavysis.processes import *

# Cell 2: Initialize
proj = Project("./my_project")
proj.import_experiments()

# Cell 3: Config
proj.update_configs("./default_config.json", overwrite="user")

# Cell 4: Process
proj.format_vid(overwrite=True)

# Cell 5: Analyze
# ... etc
```

!!! tip "Why Notebooks?"
    - Visualize progress
- Experiment interactively
- Save and share workflows
- Add markdown documentation

---

## Using behavysis Scripts

behavysis includes command-line utilities:

```bash
# Initialize a new project folder structure
behavysis-init my_project

# Launch GUI project manager
behavysis-project-gui

# Launch behavior viewer
behavysis-viewer ./my_project

# (Advanced) DLC model builder
behavysis-make-dlc-builder
```

---

## Tips for Efficient Workflows

### 1. Process in Stages

Don't run everything at once. Process a few videos first to verify settings:

```python
# Test with subset first
test_exps = proj.experiments[:3]  # First 3 experiments

for exp in test_exps:
    exp.format_vid(overwrite=True)
    # Check results before continuing
```

### 2. Use Overwrite=False to Resume

Skip completed steps:

```python
# Won't reprocess videos that already exist
proj.format_vid(overwrite=False)
proj.run_dlc(overwrite=False)
```

### 3. Check Diagnostics

After each major step, review the diagnostics:

```python
import pandas as pd

# Check what happened
diag_df = pd.read_csv("./my_project/0_diagnostics/format_vid.csv")
print(diag_df)
```

### 4. Monitor GPU Usage

For long-running DLC jobs:

```bash
# Watch GPU usage
watch -n 1 nvidia-smi
```

---

## Project Organization

### Recommended Structure

```
my_project/
├── notebooks/
│   └── analysis.ipynb          # Your analysis notebooks
├── scripts/
│   └── run_pipeline.py         # Automated scripts
├── configs/
│   └── default_config.json     # Your configuration
├── my_project/                 # behavysis project folder
│   ├── 0_configs/
│   ├── 1_raw_vid/
│   └── ...
└── results/                    # Exported results
    └── csv_exports/
```

### Keeping Separate Configs

For different analysis types:

```
configs/
├── open_field_config.json      # For open field tests
├── social_config.json          # For social interaction
└── homecage_config.json        # For home cage monitoring
```

Apply as needed:

```python
# For open field experiments
proj.update_configs("./configs/open_field_config.json", overwrite="user")
```

---

## Common Patterns

### Pattern 1: Batch Processing

```python
# Process many experiments efficiently
proj.nprocs = 8  # Use 8 CPU cores
proj.format_vid(overwrite=True)
```

### Pattern 2: Single Experiment

```python
# Work with one specific experiment
exp = proj.get_experiment("mouse_A_day1")

# Run specific steps
exp.format_vid(overwrite=True)
exp.run_dlc(gputouse=0, overwrite=True)
```

### Pattern 3: Sequential Analysis

```python
# Run one step at a time, checking each
proj.format_vid(overwrite=True)
# Check formatted videos...

proj.run_dlc(overwrite=True)
# Check DLC outputs in 3_keypoints/...

proj.calculate_parameters((...))
# Review auto-calculated values...
```

---

## Troubleshooting

### "No module named 'behavysis'"

Make sure the environment is activated:

```bash
conda activate behavysis
which python  # Should show behavysis env path
```

### CUDA out of memory

Reduce batch size or use fewer videos:

```python
# Process in smaller batches
for exp_batch in [proj.experiments[i:i+5] for i in range(0, len(proj.experiments), 5)]:
    # Process batch
    pass
```

### Video files not found

Check the folder structure:

```python
import os
raw_vid_dir = os.path.join(proj.root_dir, "1_raw_vid")
print(f"Videos found: {os.listdir(raw_vid_dir)}")
```

---

## Next Steps

- [Complete Analysis Example](../examples/analysis.md)
- [Configuration Guide](../tutorials/configs_json.md)
- [Training Behavior Classifiers](../examples/train.md)
