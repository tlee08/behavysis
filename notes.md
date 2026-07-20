Now I have a thorough understanding of the codebase and the research landscape. Here's my assessment and recommendation.

---

## Assessment & Recommendation

### The Core Architectural Problem

Your `extract_features` stage is a single hardcoded function with no plugin protocol — unlike `preprocess`, `analyse`, and `calculate_parameters` which are extensible. The generic feature battery (pairwise distances, movement sums, hull stats, rolling aggregates) is designed for **social interaction in top-down view** — it has no concept of bodypart semantics, verticality, or directional movement. For rearing and hind-paw flinch, you need **vertical/gravitational features** and **high-rate transient kinematics** that the current system doesn't compute.

### What You Can Leverage from behavysis

| Component                                        | Reusable? | Notes                                                                      |
| ------------------------------------------------ | --------- | -------------------------------------------------------------------------- |
| Pipeline orchestration (`Experiment`, `Project`) | Yes       | Stage structure, `@trace` logging, config/metadata I/O                     |
| `KEYPOINTS_SCHEMA` long-form format              | Yes       | Single mouse = `individual: "single"`, 17 bodyparts                        |
| `SubfuncModel` pattern (extra="allow")           | Yes       | For extending `ExtractFeaturesConfig`                                      |
| `PreprocessFunc` protocol                        | Yes       | Perfect for the camera sync step                                           |
| Classifier training pipeline                     | Yes       | `ClassifierContract`, `TrainingRecipe`, model registry                     |
| `read_df`/`write_df` schema validation           | Partially | Used for keypoints I/O, features bypass validation                         |
| Generic features (`compute_features`)            | Minimal   | Some base features (movement sums, centroid) still useful as supplementary |
| Constants, schemas, logging utils                | Yes       | Infrastructure                                                             |

### What You Need to Build

Three new modules and one config extension:

---

#### 1. Camera Sync Preprocess Step: `funcs/preprocess/sync_stereo.py`

**The problem**: Two synced cameras producing separate `KEYPOINTS_SCHEMA` DataFrames. The mouse can face either camera; paws are visible in at least one view.

**The approach**: A `PreprocessFunc` that merges two camera views per frame by selecting the higher-likelihood detection for each bodypart:

```
for each (frame, bodypart):
    if camA.likelihood > camB.likelihood:
        use camA (x, y, likelihood)
    else:
        use camB (x, y, likelihood)
```

This requires:

- An `individual` column that tags camera identity (e.g. `"cam_a"`, `"cam_b"`) — or two separate experiment directories
- Knowledge of which camera is which (left vs right side of cylinder)
- Frame alignment (assuming same FPS and simultaneous start trigger)

**Config model**:

```python
class SyncStereoConfig(BaseModel):
    camera_a_experiment: str  # experiment name for camera A
    camera_b_experiment: str  # experiment name for camera B
    output_individual: str = "single"
    frame_offset: int = 0  # if cameras have a known offset
```

**Output**: Standard `KEYPOINTS_SCHEMA` DataFrame with `individual="single"` and best-view coordinates.

#### 2. Behavior-Specific Feature Extraction: `funcs/extract_features/extract_features_cylinder.py`

Create a new feature module (parallel to `extract_features.py`) with semantic bodypart knowledge. The current system treats all bodyparts as interchangeable — you need to know that `nose` is "top", `hind_toe_r` is "bottom-right paw", etc. The key features:

**Rearing Features** (vertical posture):
| Feature | Formula | Rationale |
|---|---|---|
| `body_height` | `nose_y - hind_heel_mean_y` | Vertical extent of body |
| `body_length` | `nose_x - tail_base_x` (or Euclidean) | Horizontal extent |
| `elongation_ratio` | `body_height / body_length` | Rearing = ratio increases sharply |
| `back_angle_from_horizontal` | `atan2(mid_back_y - lower_back_y, mid_back_x - lower_back_x)` | Back orientation relative to ground |
| `nose_vertical_velocity` | `diff(nose_y)` per frame | Rate of rise during rearing onset |
| `front_paw_elevation` | `front_toe_mean_y - hind_heel_mean_y` | Front paws lifting |
| `centroid_y_movement` | Per-frame change in body-centroid Y | Gross vertical movement |

**Hind Paw Flinch Features** (rapid transient withdrawal):
| Feature | Formula | Rationale |
|---|---|---|
| `hind_paw_vertical_speed` | `|diff(hind_toe_mean_y)|` per frame | Sudden paw lift |
| `hind_paw_acceleration` | `diff(hind_paw_speed)` | Rate of speed change — flinch = spike |
| `hind_paw_jerk` | `diff(hind_paw_acceleration)` | 3rd derivative — captures onset sharpness |
| `hind_paw_heel_toe_distance` | Euclidean(heel, toe) per paw | Paw extension/retraction during spasm |
| `hind_paw_to_body_centroid_dist` | Euclidean(paw, centroid) | Paw withdrawal from body |
| `hind_knee_angle` | `angle3pt(hip, knee, heel)` | Leg joint angle change during flinch |
| `hind_paw_spectral_power_hf` | FFT of paw_y in rolling window (10-30 Hz band) | Tremor/spasm oscillation frequency |
| `hind_paw_asymmetry` | `abs(L_paw_speed - R_paw_speed)` | Unilateral flinch detection |

**Why jerk (3rd derivative) matters**: Research (PMC6724534, eLife 63720) shows that pain-related flinches are characterized by sub-second onset transients. The 3rd derivative (rate of change of acceleration) uniquely captures the "jerkiness" that distinguishes a flinch from normal locomotion or grooming.

**Config model**:

```python
class CylinderExtractFeaturesConfig(BaseModel):
    individuals: list[str]  # ["single"]
    bodyparts: list[str]  # all 17
    angles: list[tuple[str, str, str]]
    # Behavior-specific toggles
    compute_rearing_features: bool = True
    compute_paw_flinch_features: bool = True
    roll_windows_ms: list[int] = [33, 66, 100, 200, 500]  # research-backed windows
```

#### 3. A Key Design Decision: X-Coordinate Normalization

Since there are two cameras facing each other, the x-axis direction is **reversed** between them (left in camera A = right in camera B). After the likelihood-based merge in step 1, you may get inconsistent x-coordinates if the "best" camera switches during a bout.

**Solution**: Add a `camera_source` tracking column during merge, and optionally:

- **Flip one camera's x-coordinates** before merging (recommended, simpler)
- Or: Use a coordinate system relative to the cylinder center (requires cylinder detection)

I recommend **flipping camera B's x-coordinates** (`x = frame_width - x`) before the likelihood merge, so both views share a consistent coordinate frame.

#### 4. Integration with the Classifier

The classifier pipeline expects:

1. A wide features DataFrame (frame-indexed, float64 columns)
2. A `ClassifierContract` specifying `behaviour_name`, `individuals`, `bodyparts`, `angles`
3. Training data in `training_data/5_features_extracted/`

This part works as-is. You'd:

1. Run the new feature extractor → wide features parquet
2. Annotate frames with rearing/flinch labels
3. Train using `TrainingRecipe` → `train_all_models()` → `promote_best()`
4. Write `ClassifierContract` binding the feature space
5. Deploy through the standard `classify_behaviour` stage

### Recommended Implementation Order

```
Phase 1: Camera sync preprocess
   └── funcs/preprocess/sync_stereo.py
   └── SyncStereoConfig model

Phase 2: Extend ExtractFeaturesConfig for behavior-specific toggles
   └── CylinderExtractFeaturesConfig (or extend existing)

Phase 3: New feature extraction module
   └── funcs/extract_features/extract_features_cylinder.py
   └── Rear in = verticality, elongation, angles
   └── Hind paw flinch = kinematics (speed, accel, jerk), paw-body geometry

Phase 4: Training & deployment
   └── Annotate → train classifier → contract → predict
```

### What to Change vs What to Add

| File                                                  | Action                                    | Reason                                                               |
| ----------------------------------------------------- | ----------------------------------------- | -------------------------------------------------------------------- |
| `models/experiment_config.py`                         | **Edit** — extend `ExtractFeaturesConfig` | Add behavior toggles (`compute_rearing`, `compute_paw_flinch`)       |
| `funcs/extract_features/extract_features.py`          | **Keep** — add import of new module       | Optionally call cylinder features in `compute_features`              |
| `funcs/extract_features/extract_features_cylinder.py` | **Create**                                | New behavior-specific feature computation                            |
| `funcs/preprocess/sync_stereo.py`                     | **Create**                                | Camera merge preprocess step                                         |
| `funcs/preprocess/_helper.py`                         | **No change**                             | `PreprocessFunc` protocol already supports new functions             |
| `pipeline/experiment.py`                              | **No change**                             | `extract_features()` calls the single entry point which can delegate |

### Rating the Approach

**Strengths**:

- Fits into the existing pipeline without refactoring the protocol layer
- Camera sync as a preprocess step is architecturally clean (keypoints → keypoints)
- The classifier training pipeline is fully reusable
- Behavior-specific features are additive — you can still use generic features as supplementary signals

**Risks/limitations**:

- Likelihood-based camera merge loses information when both views are good (no 3D triangulation)
- X-coordinate flipping assumes symmetric camera placement — needs validation
- For hind paw flinch, 30-60 fps may miss sub-100ms transients; consider whether your video frame rate is sufficient
- The `extract_features` stage remaining non-pluggable means you can't easily swap feature extractors per experiment

**If you later need 3D triangulation**: DLC supports multi-camera 3D via `deeplabcut.create_multianimal_training_dataset` with calibration. That would be a larger architectural change (preprocessing step + new schema with z-coordinate). For now, 2D likelihood merge is the pragmatic 80/20 solution.

---

Shall I proceed with implementation, or would you like to discuss any part of this plan first?
