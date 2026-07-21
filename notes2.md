Now I have the full picture. Let me present the design.

---

## Proposed Design: Named Feature Sets

### The root problem

`experiment.py:206` calls a single hardcoded `extract_features()` → stores one parquet file. Every classifier in `classify_behaviour()` reads that same file. There's no way for a classifier to say "I was trained on HPW features, not generic features." This breaks as soon as you have >1 feature extractor.

### The pattern to follow

The pipeline already uses **tuple-of-plugin-functions** for `calculate_parameters`, `preprocess`, and `analyse`. Feature extraction is the odd one out. We should align it — but since outputs must be addressable per-classifier later, we need **named** extractors, not an anonymous tuple.

### Storage layout

```
5_features_extracted/
    generic/
        exp1.parquet
        exp2.parquet
    hpw/
        exp1.parquet
        exp2.parquet
```

One subdirectory per named feature set. Each experiment's parquet lives inside. (Grows naturally: add `5_features_extracted/olfactory/`, `5_features_extracted/grooming/`, etc.)

### Config changes

**`ExperimentConfig`:**

```yaml
# CURRENT (single extractor, no naming)
extract_features:
  individuals: [mouse_1, mouse_2]
  bodyparts: [nose, ear_r, ...]
  angles: [[nose, ear_r, ear_l], ...]

# PROPOSED (dict of named extractor configs)
extract_features:
  generic:
    individuals: [mouse_1, mouse_2]
    bodyparts: [nose, ear_r, ear_l, ...]
    angles: [[nose, ear_r, ear_l], ...]
  hpw: {}   # HPW is self-contained (hardcoded bodyparts), no extra params needed
```

**`ClassifyBehaviourConfig`** adds `feature_set`:

```yaml
classify_behaviour:
  - contract_fp: path/to/social_classifier/contract.yaml
    feature_set: generic # <-- NEW
    sub_behaviour: [approach, attack]
  - contract_fp: path/to/hpw_classifier/contract.yaml
    feature_set: hpw # <-- NEW
    sub_behaviour: []
```

`ClassifierContract` (in `config.py`) also gets `feature_set: str` so the contract is self-documenting about what features it was trained on.

### Feature extractor registry

`funcs/extract_features/__init__.py` gains a registry (same plugin pattern as `PreprocessFunc`):

```python
from .extract_features import extract_features as _extract_generic
from .hpw_extract_features import compute_hpw_features as _extract_hpw

FEATURE_EXTRACTORS: dict[str, Callable[..., pl.DataFrame]] = {
    "generic": _extract_generic,
    "hpw": _extract_hpw,
}
```

### Pipeline orchestration (`experiment.py`)

**`extract_features()`** — iterate named configs, call registered extractor, store in subdirectory:

```python
def extract_features(self, *, overwrite: bool) -> None:
    feature_sets = self.read_config().require_extract_features()  # returns dict[str, config]
    for name, cfg in feature_sets.items():
        out_fp = self.get_features_fp(name)
        if not overwrite and out_fp.exists():
            continue
        keypoints_df = read_df(self.get_fp(PREPROCESSED_DIR), KEYPOINTS_SCHEMA)
        extractor = FEATURE_EXTRACTORS[name]
        features_df = extractor(keypoints_df, config=self.read_config(), metadata=self.read_metadata())
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        features_df.write_parquet(out_fp)
```

**`classify_behaviour()`** — load per-classifier features, merge predictions:

```python
def classify_behaviour(self, *, overwrite: bool) -> None:
    all_predictions = []
    for model_config in self.read_config().require_classify_behaviour():
        feature_set = model_config.feature_set
        features_df = pl.read_parquet(self.get_features_fp(feature_set))
        predictions = classify_behaviour_single(model_config.contract_fp, features_df)
        all_predictions.append(predictions)
    write_df(pl.concat(all_predictions), self.get_fp(BEHAVIOUR_PREDICTED_DIR), ...)
```

**New helper** on `Experiment`:

```python
def get_features_fp(self, feature_set: str) -> Path:
    return self.root_dir / FEATURES_EXTRACTED_DIR / feature_set / f"{self.name}.{DF_IO_FORMAT}"
```

### Touchpoint summary

| File                                    | Change                                                                                 |
| --------------------------------------- | -------------------------------------------------------------------------------------- |
| `models/experiment_config.py`           | `extract_features` → `dict[str, dict]`, `ClassifyBehaviourConfig` + `feature_set: str` |
| `models/behaviour_classifier_config.py` | `ClassifierContract` + `feature_set: str`                                              |
| `funcs/extract_features/__init__.py`    | Add `FEATURE_EXTRACTORS` registry                                                      |
| `pipeline/experiment.py`                | Rewire `extract_features()`, `classify_behaviour()`, add `get_features_fp()`           |
| `pipeline/project.py`                   | Minor passthrough (already delegates to `Experiment.method`)                           |
| `constants/pipeline.py`                 | No structural changes needed                                                           |

### Backward compat

Existing `extract_features: {individuals: [...], bodyparts: [...]}` won't parse with the new schema. Migration: rename to `extract_features: {generic: {individuals: [...], bodyparts: [...], angles: [...]}}`. Since there are no CI/CD pipelines and no production users, a clean break is acceptable.

What do you think? Any concerns about this direction before I flesh out the implementation plan?
