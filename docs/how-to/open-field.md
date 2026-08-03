# Open Field — Single Mouse

Walkthrough for a standard open field experiment with one mouse.

## 1. Create the project

```bash
behavysis-make-project --preset open_field_single
```

## 2. Edit `default_config.yaml`

Open the config file. The preset has everything pre-filled except:

| Field                                              | What to set                                        |
| -------------------------------------------------- | -------------------------------------------------- |
| `run_dlc.model_fp`                                 | Absolute path to your DLC model's `config.yaml`    |
| `calculate_parameters.px_per_mm.dist_mm`           | Real-world distance between `pt_a` and `pt_b` (mm) |
| `calculate_parameters.stop_frame_from_dur.dur_sec` | Experiment duration (seconds)                      |

If your DLC model uses different bodypart names, update the anchor lists at the
top of the config. The anchors are reused throughout — change once:

```yaml
_bpts_centre: &bpts_centre [BodyCentre, TailBase1] # your centre bodyparts
```

## 3. Add videos

```bash
cp /path/to/experiment_videos/*.mp4 1_raw_videos/
```

Each `.mp4` file becomes one experiment. The filename (without `.mp4`) is the
experiment name — all stage outputs use the same name.

## 4. Run the pipeline

```bash
marimo edit run_pipeline.py
```

Run cells in order:

1. **Import & setup** — loads Project, discovers experiments
2. **Update config** — applies `default_config.yaml` to all experiments
3. **Format video** — resamples to target fps/resolution (stage 2)
4. **Run DLC** — keypoint tracking on GPU (stage 3)
5. **Calculate parameters** — auto-computes start/stop frames, px_per_mm
6. **Preprocess** — trims, interpolates low-likelihood points (stage 4)
7. **Analyse** — distance, thigmotaxis (stage 8), then combines & collates

## 5. Results

After the pipeline completes:

- `8_analysis/distance/binned_30/` — time-binned distance stats per experiment
- `8_analysis/__ALL_binned_30.parquet` — all experiments combined
- `9_analysis_combined/` — per-experiment combined analysis tables

Open `.parquet` files with `polars.read_parquet()` or export to CSV:

```python
proj.export2csv("9_analysis_combined", "./csv_output", overwrite=True)
```
