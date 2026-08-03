# Install & First Run

## Install

**1.** Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

**2.** Install the behavysis environment:

```zsh
conda env create -f path/to/conda_env.yaml
```

**3.** Install DEEPLABCUT (for keypoint tracking):

```zsh
conda activate behavysis
behavysis-init
```

**4.** Verify:

```zsh
conda activate behavysis
python -c "import behavysis"
```

## Create a project

```zsh
behavysis-make-project
```

Choose a preset. If you're unsure, pick `open_field_single`.

This creates:

```
.
├── default_config.yaml     ← edit this
├── run_pipeline.py         ← open this
├── 0_config/
├── 1_raw_videos/           ← put .mp4 files here
├── 2_formatted_videos/
├── 3_keypoints/
├── 4_preprocessed/
├── 5_features_extracted/
├── 6_behaviour_predicted/
├── 7_behaviour_scored/
├── 8_analysis/
└── 9_analysis_combined/
```

## Edit the config

Open `default_config.yaml`. Set at minimum:

- **`run_dlc.model_fp`** — absolute path to your DLC model's `config.yaml`
- **`calculate_parameters.px_per_mm.dist_mm`** — real-world arena dimension in mm

All fields are documented inline. See the `base` preset for the full reference.

## Add videos

Copy experiment `.mp4` videos into `1_raw_videos/`. Each video is one experiment.

## Run

Open `run_pipeline.py` in Jupyter Lab, VS Code, or with `marimo edit`:

```zsh
marimo edit run_pipeline.py
```

Run each cell top-to-bottom. Each cell runs one pipeline stage and saves output
to the numbered folders.

## Updating

```zsh
conda activate behavysis
conda env update -f path/to/conda_env.yaml --prune
```
