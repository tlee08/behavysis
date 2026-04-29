# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, Test, and Development Commands

```bash
# Install dependencies (uses uv)
uv sync

# Run tests
uv run pytest

# Run a single test file
uv run pytest test/test_pipeline_run.py

# Run linting
uv run ruff check src/

# Run linting with auto-fix
uv run ruff check --fix src/

# Build documentation (if needed)
uv run mkdocs serve
```

## Project Structure

Behavysis is a behavioral analysis pipeline for lab mice using computer vision. It processes video footage through DeepLabCut pose estimation and behavioral classification.

### Core Architecture

```
src/behavysis/
├── pipeline/          # Orchestration layer
│   ├── project.py     # Project: batch processing across experiments
│   └── experiment.py  # Experiment: single-experiment processing
├── processes/         # Individual processing steps (stateless functions)
├── df_classes/        # DataFrame schemas with DFMixin for IO/validation
├── models/            # Pydantic configuration models
├── behav_classifier/  # Behavioral classification (training/inference)
├── viewer/            # PySide6 GUI for behavior scoring
└── utils/             # Utilities (logging, IO, multiprocessing)
```

### Key Patterns

**Pipeline Orchestration**: `Project` manages multiple `Experiment` instances. Both expose methods that delegate to functions in `processes/`. Results are tracked via `ProcessResultCollection`.

**Processing Pattern**: Methods in `Experiment` and `Project` accept tuples of callables. Each function in `processes/` is stateless—configs and file paths are passed explicitly.

```python
# Example: running preprocess steps
exp.preprocess(
    (Preprocess.start_stop_trim, Preprocess.interpolate, Preprocess.refine_ids),
    overwrite=True
)
```

**DataFrame Classes**: All DataFrame handlers inherit from `DFMixin` in `utils/df_mixin.py`. This enforces schema validation via `IN` (index names) and `CN` (column names) enums. Default IO format is Parquet.

**Configuration**: `ExperimentConfigs` (Pydantic model) has three sections:
- `user`: User-specified settings
- `auto`: Auto-calculated values (fps, start_frame, etc.)
- `ref`: Reference values (bodypart lists) referenced via `"--ref_name"` strings

### Processing Pipeline Stages

Experiments progress through numbered folders (see `constants.py`):
1. `0_configs` - JSON configuration
2. `1_raw_vid` → `2_formatted_vid` - Video formatting
3. `2_formatted_vid` → `3_keypoints` - DeepLabCut pose estimation
4. `3_keypoints` → `4_preprocessed` - Keypoint preprocessing
5. `4_preprocessed` → `5_features_extracted` - Feature extraction for classifier
6. `5_features_extracted` → `6_predicted_behavs` - Behavioral classification
7. `6_predicted_behavs` → `7_scored_behavs` - Manual verification
8. `7_scored_behavs` → `8_analysis` - Analysis and aggregation

### Behavioral Classifier

`BehavClassifier` handles training and inference. Models are stored in `proj_dir/behav_models/<behav_name>/`. Each model has:
- `configs.json` - Model configuration
- `classifiers/<clf_struct>/` - Trained model and preprocessing pipeline
- `evaluation/` - Training history and performance metrics

### Entry Points

Defined in `pyproject.toml`:
- `behavysis-init` - Initialize new project
- `behavysis-make-project` - Create project structure
- `behavysis-project-gui` - Launch GUI
- `behavysis-viewer` - Behavior annotation viewer
- `behavysis-make-dlc-builder` - DLC model builder

### Logging

Use `logging.getLogger(__name__)` after `setup_logging()` is called at import time. Logs go to console (INFO+) and `~/.behavysis/<project>/debug.log` (DEBUG+).
