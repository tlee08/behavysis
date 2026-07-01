# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## You are...

You are a principle data scientist and data engineer.
Be critical, verify what you do, be elegant in your solutions, be honest and harsh but fair.
Always use karpathy guidelines skill.
Use context7 to search API docs (e.g. polars).
Use tavily to search the web (e.g. get up-to-date info).

## Build, Test, and Development Commands

```bash
# Install dependencies (uses uv)
uv sync

# Run linting
uv run ruff check src/

# Run linting with auto-fix
uv run ruff check --fix src/

# Build documentation
uv run mkdocs serve
```

## Project Overview

Behavysis is a behavioural analysis pipeline for lab mice using computer vision. It processes video footage through DeepLabCut pose estimation and behavioural classification.

## Core Architecture

### Pipeline Orchestration (`pipeline/`)

**Project** manages batch processing across multiple **Experiment** instances. Both expose methods that delegate to stateless functions in `processes/`. The pattern:

```python
# Experiment methods accept tuples of callables
exp.preprocess(
    (Preprocess.start_stop_trim, Preprocess.interpolate, Preprocess.refine_ids),
    overwrite=True
)
```

Each method uses `_run_funcs_with_filtered_kwargs()` which inspects function signatures and passes only the kwargs each function accepts. This allows callers to pass all available kwargs without matching exact signatures.

### DataFrame Classes (`df_classes/`)

All DataFrame handlers inherit from `DFMixin` in `utils/df_mixin.py`. Key patterns:

- Define `IN` (index names) and `CN` (column names) as Enum classes
- `read()` / `write()` handle Parquet by default (set by `DF_IO_FORMAT`)
- `basic_clean()` validates schema and sorts indices
- `check_df()` raises on schema mismatch

### Configuration Model (`models/experiment_config.py`)

`ExperimentConfig` is a Pydantic model with three sections:

- `user`: User-specified settings (formatting, preprocessing params)
- `auto`: Auto-calculated values (fps, start_frame, px_per_mm)
- `ref`: Reference values referenced via `"--ref_name"` strings

Use `config.get_ref(val)` to resolve reference strings. Use `config.get_analysis_config()` to get validated analysis parameters.

### Processing Functions (`processes/`)

Each function is stateless—config and file paths are passed explicitly. Standard signature:

```python
def some_process(
    src_fp: Path,
    dst_fp: Path,
    config_fp: Path,
    *,
    overwrite: bool,
) -> None:
```

Functions check `overwrite` first, read config with `ExperimentConfig.model_validate_json()`, process data, and write output.

## Pipeline Stages (from `constants/pipeline.py`)

```
0_config → 1_raw_vid → 2_formatted_vid → 3_keypoints → 4_preprocessed →
5_features_extracted → 6_predicted_behavs → 7_scored_behavs → 8_analysis → 9_analysis_combined
```

## Behavioural Classifier (`behav_classifier/`)

`BehavClassifier` handles training and inference. Models stored in `proj_dir/behav_models/<behav_name>/`. Key methods:

- `pipeline_training()` — trains model and saves evaluation
- `pipeline_inference(x_df)` — runs prediction on features DataFrame
- `create_from_project(proj)` — factory for all behaviours in project

## Entry Points

Defined in `pyproject.toml`:

- `behavysis-init` — Initialize new project
- `behavysis-make-project` — Create project structure
- `behavysis-project-gui` — Launch GUI
- `behavysis-viewer` — Behaviour annotation viewer
- `behavysis-make-dlc-builder` — DLC model builder

## Logging

Call `configure_logger()` once at import time (done in `__init__.py`). Use `logging.getLogger(__name__)` in modules. Logs go to console (INFO+) and `~/.behavysis/debug.log` (DEBUG+).

## Code Style

Ruff is configured with Google-style docstrings. Ignored rules include print statements (T201), broad exceptions (BLE001), and missing type annotations for `*args`/`**kwargs`.
