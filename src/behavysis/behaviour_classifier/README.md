# Behaviour Classifier

Module for training, versioning, comparing, and deploying binary behavioural classifiers. One model per behaviour. Multiple model types (RF, XGBoost, DNN variants, CNN variants) per behaviour.

## Directory structure

```
{my_beheaviour_classifier}/         # e.g. "attack", "huddling"
  training_data/                    # shared pool of labelled data (mirrors project dirs)
  production.yaml                   # {model_type, version} currently deployed
  leaderboard.yaml                  # auto-generated cross-model_type comparison
  {model_type}/                     # e.g. "rf", "dnn1", "cnn2"
    config.yaml                     # human-authored training recipe
    active.yaml                     # {version} pointer — best version for this model_type
    versions/
      {version}/                    # e.g. "v003_2025-07-07T120000"
        model.joblib                # sklearn adapter (estimator + scaler)
        model.pt                    # torch state_dict
        scaler.joblib               # MinMaxScaler (torch only)
        metadata.yaml               # resolved hyperparams + eval summary
        dataset_manifest.yaml       # dataset hash + split experiment IDs
        evaluation/                 # full eval artifacts (plots, reports)
```

## File-by-file roles

### `config.yaml` (per model_type)

Human-authored recipe. Edited by hand before training. Never auto-modified.

| Field             | Type      | Description                                                     |
| ----------------- | --------- | --------------------------------------------------------------- |
| model_type        | str       | Key in MODEL_REGISTRY (redundant with dir, but self-describing) |
| behaviour_name    | str       | Behaviour this model classifies                                 |
| individuals       | list[str] | Animal IDs used in training                                     |
| bodyparts         | list[str] | Bodyparts used for features                                     |
| seed              | int       | Random seed (default 42)                                        |
| oversample_ratio  | float     | Target pos/neg ratio for oversampling (default 0.2)             |
| undersample_ratio | float     | Target pos/neg ratio for undersampling (default 0.4)            |
| test_split        | float     | Fraction of data held out for testing (default 0.2)             |
| val_split         | float     | Fraction of training data held out for validation (default 0.2) |
| batch_size        | int       | Training batch size (default 256)                               |
| epochs            | int       | Training epochs (default 100)                                   |
| pcutoff           | float     | Probability threshold for binary prediction (default 0.2)       |

Serialisation: YAML via Pydantic `model_dump` / `model_validate`.

### `metadata.yaml` (per version)

Machine-written at the end of training. Never hand-edited.

```yaml
version: v003_2025-07-07T120000
framework: sklearn
model_type: rf
created_at: 2025-07-07T12:00:00
resolved:
  seed: 42
  batch_size: 256
  epochs: 100
  oversample_ratio: 0.2
  undersample_ratio: 0.4
  test_split: 0.2
  val_split: 0.2
data:
  n_samples: 50000
  n_features: 132
  n_train: 32000
  n_val: 8000
  n_test: 10000
  train_pos_ratio: 0.08
  test_pos_ratio: 0.07
training:
  duration_seconds: 45.2
evaluation:
  train_accuracy: 0.96
  train_f1_behav: 0.89
  val_accuracy: 0.94
  val_f1_behav: 0.87
  test_accuracy: 0.93
  test_f1_behav: 0.86
```

### `dataset_manifest.yaml` (per version)

Answers "what exactly did this version see?" Snapshot reference, not a data copy.

```yaml
version: v003_2025-07-07T120000
dataset_hash: null # hash of training_data/ state (future)
train_ids: [exp_001, exp_003, exp_004]
val_ids: [exp_002]
test_ids: [exp_005, exp_006]
n_train: 32000
n_val: 8000
n_test: 10000
```

### `active.yaml` (per model_type)

Pointer to the currently trusted version for this model_type.

```yaml
version: v003_2025-07-07T120000
promoted_at: 2025-07-07T12:02:00Z
```

Updated by auto-promotion after training (always promote if better) or by `promote_to_best()` function.

### `leaderboard.yaml` (behaviour level)

Regenerated cross-model_type comparison. Reads every model_type's `active.yaml` → version's `metadata.yaml` eval summary. Never hand-edited.

```yaml
behaviour_name: attack
generated_at: 2025-07-07T13:05:00Z
rankings:
  - model_type: rf
    version: v003_2025-07-07T120000
    test_f1_behav: 0.91
    test_accuracy: 0.95
    train_f1_behav: 0.96
    overfit_ratio: 0.05
  - model_type: dnn1
    version: v002_2025-07-06T090000
    test_f1_behav: 0.89
    test_accuracy: 0.93
    train_f1_behav: 0.97
    overfit_ratio: 0.08
```

Ranked by `test_f1_behav` descending.

`overfit_ratio` = `(train_f1_behav - test_f1_behav)` — smaller is better.

### `production.yaml` (behaviour level)

The single file inference code reads.

```yaml
model_type: rf
version: v003_2025-07-07T120000
promoted_at: 2025-07-07T13:10:00Z
```

Deliberately separate from `leaderboard.yaml` — regenerating the leaderboard never silently changes production.

## Lifecycle

### 1. Train a new version

`train(proj_dir, behaviour_name, model_type)`:

1. Read `{model_type}/config.yaml` (create default if absent)
2. Load features from `{proj_dir}/5_features_extracted/`
3. Load labels from `{proj_dir}/7_scored_behaviour/`
4. Align features and labels
5. Stratified three-way split: test split first, then val split from remainder
6. Instantiate adapter from `MODEL_REGISTRY[model_type]`
7. Train on train set, validate on val set
8. Evaluate on train, val, test
9. Generate version string: `v{seq}_{YYYY-MM-DD}THHMMSS`
10. Create `versions/{version}/` directory
11. Save model artifacts (framework-dependent)
12. Write `metadata.yaml`
13. Write `dataset_manifest.yaml`
14. Save evaluation artifacts to `evaluation/`
15. **Auto-promote**: if new version beats current active on `test_f1_behav`, update `active.yaml`
16. Return version string

### 2. Train all model types

`train_all_models(proj_dir, behaviour_name, **overrides)`:

- Loops over all keys in `MODEL_REGISTRY`
- Creates default config (with overrides) if `config.yaml` missing
- Calls `train()` for each
- After all trained, calls `regenerate_leaderboard()`

### 3. Promote within a model_type

`promote(proj_dir, behaviour_name, model_type, version)`:

- Validates that version exists
- Writes `active.yaml`

### 4. Promote to best (per model_type)

`promote_to_best(proj_dir, behaviour_name, model_type=None)`:

- For each model_type (or just one), scans all versions, picks best by `test_f1_behav`
- Calls `promote()` with the winner
- If `model_type` is None, does this for all model_types

### 5. Regenerate leaderboard

`regenerate_leaderboard(proj_dir, behaviour_name)`:

- For each model_type with `active.yaml`, reads that version's `metadata.yaml`
- Ranks by `test_f1_behav` descending
- Writes `leaderboard.yaml`

### 6. Promote to production

`promote_to_production(proj_dir, behaviour_name, model_type, version)`:

- Validates that `{model_type}/versions/{version}/` exists
- Writes `production.yaml`

### 7. Inference

`load(proj_dir, behaviour_name, *, model_type=None, version=None)`:

- If `model_type` and `version` are given: loads that specific version
- If only `model_type` is given: reads `active.yaml`, loads that version
- If neither: reads `production.yaml`, loads that version
- Returns `BehaviourClassifier` with `.config` (TrainingRecipe) and `.predict(features_df)`

### 8. Rollback

To roll back: call `promote(proj_dir, behaviour_name, model_type, old_version)` then `promote_to_production(proj_dir, behaviour_name, model_type, old_version)`. All version artifacts remain untouched in `versions/`.

## Version string format

`v{sequence_number}_{YYYY-MM-DD}THHMMSS`

Example: `v003_2025-07-07T120000`

Sequence numbers are per `(behaviour, model_type)` and determined by scanning existing version directories.

## Framework serialisation

| Framework | model_type examples          | Artifacts                                                          |
| --------- | ---------------------------- | ------------------------------------------------------------------ |
| sklearn   | rf, logreg                   | `model.joblib` (joblib dump of SklearnAdapter: estimator + scaler) |
| torch     | dnn1, dnn2, dnn3, cnn1, cnn2 | `model.pt` (state_dict), `scaler.joblib` (MinMaxScaler)            |

Torch loading: fresh TorchAdapter from MODEL_REGISTRY → `adapter.load_state(version_dir)` reconstructs model from state_dict + scaler.

## Data source

Training loads features from `{proj_dir}/5_features_extracted/` and labels from `{proj_dir}/7_scored_behaviour/`. These are the canonical sources. `training_data/` at the behaviour level mirrors these and will be populated in a future iteration.

## Promotion policy

- **Auto-promote on train**: always promotes if `test_f1_behav` improves
- **Manual promote**: `promote()` and `promote_to_best()` can be called independently
- Promotion never touches `production.yaml` — that's a separate, deliberate step

## Public API

```python
from behavysis.behaviour_classifier import (
    # Training
    train,
    train_all_models,
    # Promotion & lifecycle
    promote,
    promote_to_best,
    regenerate_leaderboard,
    promote_to_production,
    # Inference
    BehaviourClassifier,
    # Config models
    TrainingRecipe,
    VersionMetadata,
    DatasetManifest,
    ActivePointer,
    Leaderboard,
    ProductionPointer,
    # Registry
    MODEL_REGISTRY,
)
```
