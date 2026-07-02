# Pipeline Architecture

Diátaxis: **Explanation** — understanding-oriented background.

## The 10 stages

Behavysis processes experiments through numbered stages. Each stage reads the
previous stage's output and writes to its own folder:

| # | Stage | Folder | Input → Output |
|---|---|---|---|
| 0 | Config | `0_config/` | User-written `default_config.yaml` |
| 1 | Raw video | `1_raw_videos/` | `.mp4` files placed by user |
| 2 | Format video | `2_formatted_videos/` | `.mp4` → resampled `.mp4` |
| 3 | Keypoints | `3_keypoints/` | `.mp4` + DLC model → keypoint `.parquet` |
| 4 | Preprocess | `4_preprocessed/` | Keypoints → cleaned keypoints |
| 5 | Features | `5_features_extracted/` | Keypoints → derived features |
| 6 | Predicted behaviour | `6_behaviour_predicted/` | Features + classifier → predicted labels |
| 7 | Scored behaviour | `7_behaviour_scored/` | Predicted labels → scored (user-verified) labels |
| 8 | Analysis | `8_analysis/` | Keypoints / scored behaviours → statistical summaries |
| 9 | Combined | `9_analysis_combined/` | Per-analysis results → single combined table |

All files for experiment `exp1` are `exp1.<ext>` in their respective folders.

## Parallel execution

The `Project` class runs most stages in parallel across experiments using
[Dask](https://dask.org/) `LocalCluster`. Set `proj.nprocs` to control worker
count. Some stages force sequential execution (e.g. `classify_behaviour` due to
potential I/O conflicts with shared model files).

## `Experiment` vs `Project`

- `Experiment(name, root_dir)` — a single experimental run. Initialised by
  scanning for the experiment's files across all stage folders.

- `Project(root_dir)` — a directory of experiments. Batch-runs the same
  stage across all experiments, parallel where possible.

Most user code interacts with `Project`. `Experiment` is the unit of work.
