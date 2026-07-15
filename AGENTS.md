# AGENTS.md — Behavysis

## You: The Agent

You are a principle data scientist and data engineer.

- Use your critical thinking.
- Validate your thinking and work at every step with checks.
- Accept only elegant solutions.
- Be honest, critical, harsh, and fair.

You must always use:

- karpathy-guideline skill ALWAYS.
- context7 MCP for API and SDK information.
- tavily MCP for searching the web.
- GitHub MCP to make commits.

## Setup & environment

```bash
conda env create -f conda_env.yaml
conda activate behavysis
uv pip install -e ".[dev]"
```

The project uses **uv** as the build backend (`uv_build`). Dependencies are in `pyproject.toml`, pinned in `uv.lock`. Conda handles `ffmpeg` and `hdf5` (system-level deps). Install both layers — pip-only installs will miss `ffmpeg`.

## Dev commands

```bash
# Lint (ruff selects ALL rules, but almost every rule is currently ignored)
uv run ruff check src/

# Run all tests
uv run pytest

# Run only fast tests (exclude slow, integration, gpu)
uv run pytest -m "not slow and not integration and not gpu"

# Run a single test file
uv run pytest tests/unit/test_constants.py

# Run a single test by name
uv run pytest tests/functional/test_polars_schema.py::test_keypoints_schema_validates -x

# Serve docs locally
uv run mkdocs serve
```

There is **no CI, no pre-commit hooks, no Makefile**. Lint and test are manual.

## Architecture

An animal behaviour video-analysis pipeline: raw footage → formatted video → DLC keypoint tracking → preprocessing → feature extraction → behaviour classification → analysis → combined summary.

### Core domain: `Experiment` and `Project`

`behavysis.pipeline.experiment.Experiment` — a single experimental run (one video). Stages are numbered folders (`1_raw_videos/`, `2_formatted_videos/`, … `9_analysis_combined/`). Each stage reads the previous stage's output, writes its own.

`behavysis.pipeline.project.Project` — a directory of experiments. Orchestrates batch operations.

### Data: Polars long-form schemas (NOT pandas MultiIndex)

The project is **mid-migration** from pandas MultiIndex to Polars long-form. Schemas are defined in `src/behavysis/schemas/schemas.py` as typed dicts (e.g. `KEYPOINTS_SCHEMA`, `BEHAVIOUR_PREDICTED_SCHEMA`, `ANALYSIS_SCHEMA`). All I/O goes through `read_df`/`write_df` which validate against these schemas at read/write boundaries. Storage format: Parquet.

**Key schemas:**

- `KEYPOINTS_SCHEMA` — one row per `(frame, individual, bodypart)` with `x, y, likelihood`
- `BEHAVIOUR_PREDICTED_SCHEMA` — `(frame, behaviour, prob, pred)`
- `ANALYSIS_SCHEMA` — `(frame, individual, measure, value)`
- `SUMMARY_SCHEMA`, `BINNED_SCHEMA` — aggregated stats
- `COMBINED_ANALYSIS_SCHEMA`, `COLLATED_*` — concatenated/collated variants

`df_classes/` directory is **empty** (old pandas DFMixin classes removed during migration). `architectural_recommendations.md` documents the prior design and migration rationale — treat it as historical context, not current spec.

### Processing functions: `src/behavysis/funcs/`

Each pipeline stage maps to a submodule here. Function protocol classes (`PreprocessFunc`, `AnalyseFunc`, `CalculateParametersFunc`) use a `run(df, config) -> df` signature for plugin-like stages.

### Configuration: `src/behavysis/models/`

Pydantic v2 models for `ExperimentConfig`, `ExperimentMetadata`, `BehaviourClassifierConfig`, and `BoutStruct`. Configs are YAML files stored per-experiment in `0_config/`.

### Templates: `src/behavysis/templates/`

Jinja2 templates for project scaffolding (`run_pipeline_script.py`, DLC config, etc.). Rendered via `template_utils.py` using `PackageLoader("behavysis", "templates")`.

### Logging

Uses `loguru`. The `@trace` decorator (`utils/logger_utils.py`) logs entry/exit/duration for each pipeline stage. **IMPORTANT**: `@trace` logs them then re-raises, `pass_exception` catches them — the pipeline continues to the next experiment on failure.

### Plotting

Matplotlib configured with backend `Agg` (non-interactive, set in `__init__.py`). Seaborn theme with `whitegrid` style. `PLOT_DPI` constant controls all figure rendering.

## Testing

- **Unit tests**: `tests/unit/` — test individual functions with synthetic data
- **Functional tests**: `tests/functional/` — test schema correctness, SimBA feature extraction
- **Integration tests**: `tests/integration/` — test pipeline orchestration end-to-end
- **Fixtures**: `conftest.py` provides `keypoints_df_data` (synthetic DLC output), `multiindex_df`, `minimal_config_dict`
- **Test data**: `tests/data/` — fixture data for integration tests
- **Markers**: `slow`, `integration`, `gpu` — use `-m` to filter
- **No external services required** except GPU-marked tests

## Style conventions

- Docstrings: **numpy style** (configured in `ruff.lint.pydocstyle`)
- `from __future__ import annotations` at top of files
- Type hints everywhere (PEP 585 generics like `dict[...]`, `list[...]` instead of `typing.Dict/List`)
- Public API re-exported through `__init__.py` — import from `behavysis.schemas`, not `behavysis.schemas.schemas`
- String constants for column names live in `constants/data_names.py` (e.g. `FRAME = "frame"`, `INDIVIDUAL = "individual"`)
