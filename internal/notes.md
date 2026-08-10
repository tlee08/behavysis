# Architecture Notes & Remaining Tasks

## Named Feature Sets ✅

The pipeline supports multiple named feature extractors stored in subdirectories:

```
5_features_extracted/
    extract_generic/
        exp1.parquet
    extract_hpw/
        exp1.parquet
```

`Experiment.get_features_fp(feature_set: str)` resolves paths. `ClassifierContract.feature_set` binds a classifier to its training config. Extractors are passed as a tuple of callables to `extract_features()` — `func.__name__` becomes the feature set name.

## HPW Feature Extraction ✅

Implemented as `funcs/extract_features/extract_hpw.py`. Computes rearing (R01-R06) and hind-paw withdrawal (W01-W19) features plus cross-features (X01-X03) with rolling-window aggregates.

Config model `FeaturesHpwConfig` is empty — all features always computed, no user-facing toggles.

### Rearing features (reference)

| Feature                           | Formula                              |
| --------------------------------- | ------------------------------------ |
| `back_angle_deg`                  | Angle of back relative to horizontal |
| `nose_elevation_mm`               | Nose height above estimated floor    |
| `body_elongation_ratio`           | Height / length                      |
| `centroid_vertical_velocity_mm_s` | Per-frame change in body-centroid Y  |
| `front_paw_elevation_mm`          | Front paw height above hind heel     |
| `nose_vertical_velocity_mm_s`     | Per-frame change in nose Y           |

### Hind paw flinch features (reference)

Vertical/horizontal velocity, vy-to-vx ratio, elevation above floor, heel-toe distance, vertical acceleration, left/right asymmetry, paw vertical velocity peaks, body-stillness controls. 3rd derivative (jerk) captures sub-second onset transients characteristic of pain-related flinches (PMC6724534, eLife 63720).

## Camera Sync Preprocess ❌

**Not implemented.**

Two synced cameras produce separate `KEYPOINTS_SCHEMA` DataFrames. A `PreprocessFunc` should merge views per frame by selecting the higher-likelihood detection for each bodypart.

```python
class SyncStereoConfig(BaseModel):
    camera_a_experiment: str
    camera_b_experiment: str
    output_individual: str = "single"
    frame_offset: int = 0
```

**X-coordinate normalization**: Flip camera B's x-coordinates (`x = frame_width - x`) before merging so both views share a consistent coordinate frame.

**Risks**:

- Likelihood-based merge loses information when both views are good (no 3D triangulation)
- X-coordinate flipping assumes symmetric camera placement
- 30-60 fps may miss sub-100ms flinch transients

For full 3D triangulation later: DLC supports multi-camera 3D via `deeplabcut.create_multianimal_training_dataset` with calibration.

## Social Feature Gaps ❌

The generic feature extractor computes inter-individual distances but misses interpretable social-proximity signals:

1. **Min cross distance** (`{ind_a}_{ind_b}_cross_dist_min`) — minimum bodypart-to-bodypart distance per frame. One line from existing `_compute_cross_distances`.
2. **Speed toward/away** (`{ind_a}_{ind_b}_centroid_speed_toward`) — `diff(centroid_dist)`. Negative = approaching, positive = retreating.
3. **Movement asymmetry** (`movement_asymmetry_{ind_a}_{ind_b}`) — ratio of per-individual movement sums. High asymmetry = chasing.
4. **Inter-animal orientation** — relative angle between heading vectors. Strongest signal for directed social behaviour but requires new infrastructure.

The first three are trivial derivations of existing features.

## Bout Conversion Simplification ❌

The current behaviour-scored architecture is wide-form with dynamic columns:

```
BoutStruct → get_bouts_struct() → predicted_to_scored() → frames2bouts() → Bouts → bouts2frames()
```

`BoutStruct`, `Bouts`, `Bout` models exist solely to carry dynamic column metadata. `get_bouts_struct` runtime-scans the DataFrame to discover sub-behaviour columns. `predicted_to_scored` materializes empty columns for all sub-behaviours across all behaviours.

**Proposed**: Convert to long-form `[frame, behaviour, sub_behaviour, value]` — mirroring `ANALYSIS_SCHEMA` (`[frame, individual, measure, value]`):

```python
BEHAVIOUR_SCORED_SCHEMA: SchemaDict = {
    FRAME:         pl.Int64,
    BEHAVIOUR:     pl.Utf8,
    SUB_BEHAVIOUR:  pl.Utf8,
    VALUE:         pl.Int64,
}
```

- `predicted_to_scored` → unpivot PRED into rows, add `sub_behaviour="actual"` rows
- `frames2bouts` → filter per-behaviour, `vect2bouts`, join sub-behaviour values via `group_by`
- `bouts2frames` → explode bout ranges, cross join with sub-behaviours
- Eliminates `get_bouts_struct`, `BoutStruct`, dynamic columns entirely
- ~3-5x row count increase, negligible with Polars

Aligns with the project's established long-form pattern used by `ANALYSIS_SCHEMA`.
