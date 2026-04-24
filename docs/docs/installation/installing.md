# Installing behavysis

This guide walks you through installing behavysis and its dependencies.

!!! tip "Requirements"
    - Operating System: Linux, macOS, or Windows
    - RAM: 8 GB minimum (16 GB recommended)
    - GPU: Optional but strongly recommended for DeepLabCut
    - Storage: ~5 GB for environments + data space

---

## Step 1: Install Conda

Conda is a package manager that handles Python environments and dependencies.

=== "Linux"

    1. Download the [Miniconda installer](https://docs.conda.io/en/latest/miniconda.html)
    2. Open a terminal
    3. Run the installer:
    ```bash
    bash Miniconda3-latest-Linux-x86_64.sh
    ```
    4. Follow the prompts and restart your terminal
    5. Verify installation:
    ```bash
    conda --version
    # Should show: conda xx.xx.xx
    ```

=== "macOS"

    1. Visit the [Miniconda downloads page](https://docs.conda.io/en/latest/miniconda.html)
    2. Download the macOS installer (Intel or M1/M2)
    3. Open the downloaded file and follow the prompts
    4. Open a terminal and verify:
    ```bash
    conda --version
    # Should show: conda xx.xx.xx
    ```

=== "Windows"

    1. Visit the [Miniconda downloads page](https://docs.conda.io/en/latest/miniconda.html)
    2. Download the Windows installer
    3. Run the installer and follow the prompts
    4. Open "Anaconda PowerShell Prompt" from the Start menu
    5. Verify:
    ```powershell
    conda --version
    # Should show: conda xx.xx.xx
    ```

!!! note "Already have Anaconda?"
    You can use existing Anaconda/Miniconda installations. Just ensure `conda` is available in your terminal.

---

## Step 2: Configure Conda

Update conda and set up the fast solver:

```bash
# Update conda
conda update -n base conda

# Install fast solver (much faster!)
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

# Install Jupyter integration (optional)
conda install -n base nb_conda nb_conda_kernels
```

---

## Step 3: Install behavysis

### Option A: From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/tlee08/behavysis.git
cd behavysis

# Create the conda environment
conda env create -f conda_env.yaml

# Activate the environment
conda activate behavysis

# Install behavysis in editable mode
pip install -e .
```

### Option B: Using pip (When Available)

```bash
# Create environment manually
conda create -n behavysis python=3.12 ffmpeg hdf5 -c conda-forge
conda activate behavysis

# Install behavysis
pip install behavysis
```

---

## Step 4: Verify Installation

Test that behavysis is working:

```python
# In Python or Jupyter
import behavysis
print(behavysis.__version__)

from behavysis import Project
print("✓ behavysis imported successfully")
```

Expected output:
```
0.1.24
✓ behavysis imported successfully
```

---

## Step 5: Install DeepLabCut (Required for Pose Estimation)

DeepLabCut is a separate environment for running pose estimation.

```bash
# Download the DLC environment file
curl -O https://raw.githubusercontent.com/DeepLabCut/DeepLabCut/main/conda-environments/DEEPLABCUT.yaml

# Create the DLC environment
conda env create -f DEEPLABCUT.yaml
```

### Verify DLC Installation

```bash
# Activate DLC environment
conda activate DEEPLABCUT

# Test import
python -c "import deeplabcut; print('✓ DeepLabCut installed')"

# Return to behavysis
conda activate behavysis
```

---

## Step 6: Install SimBA (Optional)

SimBA provides additional behavior classification features.

```bash
# From within the behavysis repository directory
conda env create -f simba_env.yaml
```

Or download and create manually:

```bash
curl -O https://raw.githubusercontent.com/tlee08/behavysis/main/simba_env.yaml
conda env create -f simba_env.yaml
```

---

## Environment Summary

You'll now have three conda environments:

| Environment | Purpose | When to Use |
|:------------|:--------|:------------|
| `behavysis` | Main analysis pipeline | Most operations |
| `DEEPLABCUT` | Pose estimation | Running DLC models |
| `simba` | Behavior classification (optional) | SimBA features |

### Switching Between Environments

```bash
# Main analysis
conda activate behavysis

# To run pose estimation
conda activate DEEPLABCUT
deeplabcut

# Back to behavysis
conda activate behavysis
```

---

## GPU Support

### NVIDIA GPUs (CUDA)

If you have an NVIDIA GPU, ensure CUDA is installed:

```bash
# Check if CUDA is available
nvidia-smi

# In Python:
import torch
print(torch.cuda.is_available())  # Should be True
```

### macOS Metal (M1/M2/M3)

PyTorch on Apple Silicon uses Metal Performance Shaders (MPS):

```python
import torch
print(torch.backends.mps.is_available())  # Should be True on M-series
```

### No GPU (CPU Only)

DeepLabCut will run on CPU but expect **much** slower performance (10-50× slower).

---

## Post-Installation Setup

### Configure Jupyter (Optional)

If using Jupyter notebooks:

```bash
# Make behavysis available in Jupyter
conda activate behavysis
python -m ipykernel install --user --name behavysis --display-name "Python (behavysis)"
```

### Verify GPU Usage

```bash
# In Python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## Troubleshooting

### "conda command not found"

Your shell isn't configured for conda. Try:

```bash
# Initialize for your shell
conda init bash  # or zsh, fish, etc.

# Then restart terminal
```

### Package conflicts during installation

Try updating conda first:

```bash
conda update --all
conda env create -f conda_env.yaml --force
```

### PyTorch/CUDA version mismatch

If you get CUDA errors, reinstall PyTorch with correct CUDA version:

```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### ImportError: No module named 'behavysis'

Make sure you:
1. Activated the correct environment: `conda activate behavysis`
2. Installed with `-e .` flag (editable install)
3. Are running Python from the repository directory

---

## Next Steps

- [Running behavysis](running.md) — Start your first analysis
- [Project Setup Tutorial](../tutorials/setup.md) — Learn the workflow
- [Configuration Guide](../tutorials/configs_json.md) — Understand settings

---

## Uninstallation

If you need to remove behavysis:

```bash
# Remove conda environments
conda env remove -n behavysis
conda env remove -n DEEPLABCUT
conda env remove -n simba

# Remove repository (if installed from source)
cd ..
rm -rf behavysis
```

See [complete uninstallation guide](uninstalling.md) for details.
