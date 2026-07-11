# Behaviour Classifier

Module for training, versioning, comparing, and deploying binary behavioural classifiers. One model per behaviour. Multiple model types (RF, XGBoost, DNN variants, CNN variants) per behaviour.

## Directory structure

```
{my_behaviour_classifier}/          # arbitrary name; behaviour is set in contract.yaml
  training_data/                    # shared pool of labelled data (mirrors project dirs)
  contract.yaml                     # shared behaviour + feature contract (all model_types)
  production.yaml                   # {model_type, version} currently deployed
  leaderboard.yaml                  # auto-generated cross-model_type comparison
  {model_type}/                     # e.g. "rf", "dnn1", "cnn2"
    config.yaml                     # human-authored hyperparameter recipe
    active.yaml                     # {version} pointer — best version for this model_type
    versions/
      {version}/                    # e.g. "v003_2025-07-07T120000"
        model.joblib                # sklearn adapter (estimator + scaler + feature mask)
        model.pt                    # torch state_dict
        scaler.joblib               # MinMaxScaler (torch only)
        feature_mask.npy            # selected feature indices (torch only)
        metadata.yaml               # recipe snapshot + eval summary
        dataset_manifest.yaml       # dataset hash + split experiment IDs
        evaluation/                 # full eval artifacts (plots, reports)
```

## File-by-file roles

### `contract.yaml` (per classifier)

Human-authored shared contract — the single source of truth for what every
model_type in the classifier trains on. Edited by hand before training, never
auto-modified.

| Field          | Type      | Description                                 |
| -------------- | --------- | ------------------------------------------- |
| behaviour_name | str       | Behaviour this classifier classifies        |
| individuals    | list[str] | Animal IDs used in training                 |
| bodyparts      | list[str] | Bodyparts used for features                 |

### `config.yaml` (per model_type)

Human-authored hyperparameter recipe. Edited by hand before training. Never auto-modified.

| Field             | Type      | Description                                                     |
| ----------------- | --------- | --------------------------------------------------------------- |
| model_type        | str       | Key in MODEL_REGISTRY (redundant with dir, but self-describing) |
| seed              | int       | Random seed (default 42)                                        |
| oversample_ratio  | float     | Target pos/neg ratio for oversampling (default 0.2)             |
| undersample_ratio | float     | Target pos/neg ratio for undersampling (default 0.4)            |
| test_split        | float     | Fraction of data held out for testing (default 0.2)             |
| val_split         | float     | Fraction of training data held out for validation (default 0.2) |
| batch_size        | int       | Training batch size (default 256)                               |
| epochs            | int       | Training epochs (default 100)                                   |
| pcutoff           | float     | Probability threshold for binary prediction (default 0.2)       |
| feature_selection | bool      | Drop uninformative features, fit on train split (default True)  |
| variance_threshold| float     | Min variance (post-scaling) to keep a feature (default 0.0)     |
| max_features      | int\|null | Cap to top-k features by RF importance (default null = all)     |

Serialisation: YAML via Pydantic `model_dump` / `model_validate`.

### `metadata.yaml` (per version)

Machine-written at the end of training. Never hand-edited. `recipe` is a full
snapshot of the `config.yaml` used for this version.

```yaml
version: v003_2025-07-07T120000
framework: sklearn
model_type: rf
created_at: 2025-07-07T12:00:00
recipe:
  model_type: rf
  seed: 42
  oversample_ratio: 0.2
  undersample_ratio: 0.4
  test_split: 0.2
  val_split: 0.2
  batch_size: 256
  epochs: 100
  pcutoff: 0.2
  feature_selection: true
  variance_threshold: 0.0
  max_features: null
data:
  n_samples: 50000
  n_features: 132
  n_features_selected: 96
  n_train: 32000
  n_test: 10000
  train_pos_ratio: 0.08
  test_pos_ratio: 0.07
training:
  duration_seconds: 45.2
evaluation:
  train_accuracy: 0.96
  train_f1_behav: 0.89
  test_accuracy: 0.93
  test_f1_behav: 0.86
```

### `dataset_manifest.yaml` (per version)

Answers "what exactly did this version see?" Snapshot reference, not a data copy.

```yaml
version: v003_2025-07-07T120000
dataset_hash: null # hash of training_data/ state (future)
train_ids: [exp_001, exp_003, exp_004]
test_ids: [exp_005, exp_006]
n_train: 32000
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

### `production.yaml` (classifier level)

A pure pointer to the deployed `model_type`. Written only by
`promote_to_production` — never hand-edited. The deployed version is resolved
from that model_type's `active.yaml`; the behaviour and feature contract are
resolved from `contract.yaml`.

```yaml
model_type: rf
promoted_at: 2025-07-07T13:10:00Z
```

Deliberately separate from `leaderboard.yaml` — regenerating the leaderboard never silently changes production.

## Lifecycle

### 1. Train a new version

`train(clf_dir, model_type)`:

1. Read `contract.yaml` (shared) and `{model_type}/config.yaml` (must exist — authored, never auto-created here)
2. Load features from `{clf_dir}/training_data/5_features_extracted/`
3. Load labels from `{clf_dir}/training_data/7_behaviour_scored/`
4. Align features and labels
5. Stratified three-way split: test split first (grouped by video), then val split from remainder
6. Instantiate adapter from `MODEL_REGISTRY[model_type]`
7. Train on train set (scale → select features on train only → fit), validate on val set
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

`train_all_models(clf_dir, behaviour_name, individuals, bodyparts)`:

- Writes the shared `contract.yaml` (behaviour + feature contract) if missing
- Loops over all keys in `MODEL_REGISTRY`
- Creates a default `config.yaml` (hyperparameters) per model_type if missing
- Calls `train()` for each
- After all trained, calls `regenerate_leaderboard()`

### 3. Promote within a model_type

`promote(clf_dir, model_type, version)`:

- Validates that version exists
- Writes `active.yaml`

### 4. Promote to best (per model_type)

`promote_to_best(clf_dir, model_type=None)`:

- For each model_type (or just one), scans all versions, picks best by `test_f1_behav`
- Calls `promote()` with the winner
- If `model_type` is None, does this for all model_types

### 5. Regenerate leaderboard

`regenerate_leaderboard(clf_dir)`:

- For each model_type with `active.yaml`, reads that version's `metadata.yaml`
- Ranks by `test_f1_behav` descending
- Writes `leaderboard.yaml`

### 6. Promote to production

`promote_to_production(clf_dir, model_type)`:

- Validates that the model_type has an `active.yaml`
- Writes `production.yaml` (a pure `{model_type}` pointer)

### 7. Inference

`BehaviourClassifier.load(clf_dir, *, model_type=None, version=None)`:

- If `model_type` and `version` are given: loads that specific version
- If only `model_type` is given: reads `active.yaml`, loads that version
- If neither: reads `production.yaml` → that model_type's `active.yaml` → version
- Returns `BehaviourClassifier` with `.config` (TrainingRecipe), `.contract` (ClassifierContract), and `.predict(features_df)`

### 8. Rollback

To roll back a model_type to an earlier version: call `promote(clf_dir, model_type, old_version)` to repoint its `active.yaml`. If that model_type is in production, `production.yaml` then resolves to `old_version` automatically. All version artifacts remain untouched in `versions/`.

## Version string format

`v{sequence_number}_{YYYY-MM-DD}THHMMSS`

Example: `v003_2025-07-07T120000`

Sequence numbers are per model_type (within a classifier dir) and determined by scanning existing version directories.

## Framework serialisation

| Framework | model_type examples          | Artifacts                                                          |
| --------- | ---------------------------- | ------------------------------------------------------------------ |
| sklearn   | rf, logreg                   | `model.joblib` (joblib dump of SklearnAdapter: estimator + scaler + feature mask) |
| torch     | dnn1, dnn2, dnn3, cnn1, cnn2 | `model.pt` (state_dict), `scaler.joblib` (MinMaxScaler), `feature_mask.npy` |

Torch loading: fresh TorchAdapter from MODEL_REGISTRY → `adapter.load_state(version_dir)` reconstructs model from state_dict + scaler + feature mask (the mask length defines `nfeatures`).

## Data source

Training loads features from `{clf_dir}/training_data/5_features_extracted/` and labels from `{clf_dir}/training_data/7_behaviour_scored/`. A classifier is self-contained: assemble these folders (e.g. copy extracted features from a processed project, convert BORIS exports to scored labels) before training. See the `train_behaviour_classifier` marimo notebook for the full assemble → train → evaluate → promote flow.

## Feature selection

When `feature_selection` is enabled (default), each `fit` scales all columns with a `MinMaxScaler`, then selects features **on the training split only**:

1. `VarianceThreshold(variance_threshold)` drops near-constant columns. Because scaling happens first, the threshold is comparable across features (variance lies in `[0, 0.25]`).
2. If `max_features` is set, a random forest ranks the survivors and keeps the top-k by importance.

The selected column indices (`feature_mask`) are persisted with the model and applied identically at inference. `metadata.yaml` records both `n_features` (total extracted) and `n_features_selected`.

## Evaluation artifacts

Every trained version writes to `versions/{version}/evaluation/`:

| File | Contents |
| --- | --- |
| `{train,val,test}_report.json` | precision / recall / f1 per split |
| `{train,val,test}_eval.parquet` | per-frame prob / pred / actual |
| `{split}_confm.png` | confusion matrix |
| `{split}_pcutoffs.png` | metrics vs probability cutoff — use to choose `pcutoff` |
| `{split}_logc.png` | predicted-probability distribution |
| `history.png`, `history.parquet` | train/val loss curve (torch only) |
| `feature_importance.png` | top features by importance (sklearn) |
| `feature_report.json` | `n_features_total` vs `n_features_used` after selection |

Inspect these — plus each version's `metadata.yaml` (train/val/test F1) and the `overfit_ratio` in `leaderboard.yaml` — before promoting a model to production.

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
    ClassifierContract,
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
