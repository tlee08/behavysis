# Running

**Step 1:** Activate the environment

```zsh
conda activate behavysis
```

**Step 2:** Create a project

```zsh
behavysis-make-project
```

This scaffolds a project directory with a preset config and notebook. Choose a
preset that matches your experiment type:

| Preset | Use case |
|---|---|
| `open_field_single` | One mouse, open field — speed, freezing, thigmotaxis |
| `social_two_mice` | Two mice, social interaction — all above + social distance |
| `dlc_only` | Just keypoint tracking, no analysis |
| `behaviour_pipeline` | Full pipeline with automated behaviour classification |
| `base` | All options documented — use as a reference |

**Step 3:** Edit the config

Open `default_config.yaml` and set at minimum:

- `run_dlc.model_fp` — path to your DLC model's `config.yaml`
- `calculate_parameters.px_per_mm.dist_mm` — real-world arena size (mm)

**Step 4:** Add videos

Copy your `.mp4` experiment video(s) into `1_raw_videos/`.

**Step 5:** Run the pipeline

Open `run_pipeline.py` in Jupyter Lab, VS Code, or any marimo-compatible editor.
Run cells top-to-bottom — each cell runs one pipeline stage.

Or run the marimo web UI:

```zsh
marimo edit run_pipeline.py
```
