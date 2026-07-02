# Config System

Diátaxis: **Explanation** — understanding the config model and preset system.

## Config model

Config files are YAML validated against `ExperimentConfig` (Pydantic v2). Every
top-level key maps to a pipeline stage:

```yaml
format_video: {...}           # video resolution, fps, trim
run_dlc: {...}                # DLC model path
calculate_parameters: {...}   # auto-calculated experiment params
preprocess: {...}             # keypoint cleaning functions
extract_features: {...}       # SimBA feature extraction settings
classify_behaviour: [...]     # behaviour model list
analyse: {...}                # quantitative analysis functions
```

All seven keys are required by the validator. Set unused stages to `null`.

## Sub-function config

`calculate_parameters`, `preprocess`, and `analyse` use `SubfuncModel` — each
sub-key maps to a function from `behavysis.funcs`. Sub-keys are validated at
runtime, not at config parse time, so you can add custom analysis functions
without modifying the model.

## Preset system

Instead of writing configs from scratch, start with a preset:

```bash
behavysis-make-project --list       # see available presets
behavysis-make-project --preset open_field_single
```

Each preset is a validated `default_config.yaml` + matching `run_pipeline.py`
notebook. The `base` preset documents every field.

Presets live in `src/behavysis/presets/` and ship with the package. To see them:

```python
from behavysis.presets import list_presets, describe_presets

list_presets()        # ['base', 'behaviour_pipeline', 'dlc_only', ...]
describe_presets()    # {'base': 'Reference template ...', ...}
```

## Config validation

Every config read/write goes through schema validation:

```python
from behavysis.models import ExperimentConfig

config = ExperimentConfig.read_yaml("default_config.yaml")  # validates
config.require_run_dlc().model_fp                           # typed access
```
