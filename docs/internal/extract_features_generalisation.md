# Extract Features Generalisation — Specification (v2)

## Context

`src/behavysis/funcs/extract_features/extract_features.py` replicates the SimBA `ExtractFeaturesFrom16bps` pipeline in Polars + NumPy + SciPy. It is hardcoded to exactly 2 animals × 8 specific bodyparts. Every layer is coupled to this assumption.

SimBA itself validates our planned approach: for user-defined bodypart configurations, SimBA computes a **generic feature battery** — "the distance between all body-parts and their velocities in rolling windows" — using `combinations(bp_names, 2)`. No semantic bodypart role mapping.

## Final Design Decisions

| Decision | Choice |
|---|---|
| Feature computation | Full cartesian product of all bodypart pairs. Purely programmatic. No semantic roles. |
| Feature naming | Generic: `{indiv}_{bp_a}_to_{bp_b}_dist`, `{indiv}_{bp}_movement`, etc. |
| Backward compatibility | None. Aggressive refactor. |
| Feature scope | All generic features (pairwise distances, movements, hull, cdist, probability, rolling, deviations, percentile ranks) |
| Config location | `ExtractFeaturesConfig` (experiment config) and `ClassifyBehaviourConfig` (model config) — cross-validated at runtime |
| Coupling model↔features | Option B: model config includes `individuals` + `bodyparts`; validated against experiment's `ExtractFeaturesConfig` at classification time |
| Overfitting mitigation | Trust RF robustness (Option 1) + diagnostic reporting (Option 5): feature importance + SHAP analysis saved to model eval dir. No automated feature selection. |
| Individual labels | User-provided. "Mouse_1", "Mouse_2" etc. are gone. |
| Migration | Clean break. No deprecation path. |

## Architecture

```
ExtractFeaturesConfig:
    individuals: list[str]   # e.g. ["mouse1marked", "mouse2unmarked"]
    bodyparts: list[str]     # e.g. ["Nose", "LeftEar", "TailBase1", ...]

ClassifyBehaviourConfig:
    proj_dir: Path
    behaviour_name: str
    individuals: list[str]   # model trained on THIS bodypoint config
    bodyparts: list[str]     # model trained on THIS bodypoint config
    pcutoff: float | None
    min_empty_window_secs: float
    user_defined: list[str]
```

Runtime validation in `classify_behaviour()`:
```
assert model_cfg.individuals == experiment_cfg.individuals
assert model_cfg.bodyparts == experiment_cfg.bodyparts
```

## Generic Feature Battery

| # | Feature group | Computation | Column naming |
|---|---|---|---|
| 1 | Within-individual pairwise distances | `combinations(bodyparts, 2)` per individual. Euclidean distance / px_per_mm. | `{indiv}_{bp_a}_to_{bp_b}_dist` |
| 2 | Cross-individual pairwise distances | Cartesian product of (indiv, bp) pairs across different individuals. | `{indiv_a}_{bp_a}_to_{indiv_b}_{bp_b}_dist` |
| 3 | Per-bodypart movement | Frame-to-frame Euclidean distance per (indiv, bp) | `{indiv}_{bp}_movement` |
| 4 | Convex hull | Perimeter + areal change per individual | `{indiv}_hull_perimeter`, `{indiv}_hull_area_change` |
| 5 | Cdist statistics | Pairwise hull distances: max, min, mean, sum per individual | `{indiv}_cdist_max/min/mean/sum` |
| 6 | Total movements | Sum of all movements per individual + grand total | `total_movement_{indiv}`, `total_movement_all` |
| 7 | Total cdist | Sum of individual cdist sums | `cdist_sum_all` |
| 8 | Probability features | Sum of all likelihoods, low-prob detection counts | `sum_probabilities`, `low_prob_detections_0.1/0.5/0.75` |
| 9 | Rolling windows | median/mean/sum at windows fps/2, fps/2.5, ... for groups 1–7 | `{base_name}_median/mean/sum_{window}` |
| 10 | Deviations | mean - current for key aggregate features | `{base_name}_deviation` |
| 11 | Percentile ranks | Percentile rank for total movements and selected features | `{base_name}_percentile_rank` |

**Dropped** (semantic, no longer applicable):
- `Mouse_1_angle` / `Total_angle_both_mice` (requires nose→center→tail triplet)
- `Tortuosity_Mouse1_*` (requires designated centroid bodypart)
- `Tail_end_relative_to_tail_base_centroid_nose` (semantic combination)
- All SimBA-naming hardcoded columns (e.g. `Mouse_1_nose_to_tail`, `M1_hull_large_euclidean`, etc.)

**Future**: Angle features and tortuosity can be added as optional plugins if a config provides bodypart role annotations (e.g. `angle_triplet: [Nose, BodyCentre, TailBase1]`).

## Diagnostic Reporting (Phase 2)

Added to `BehaviourClassifier.evaluate()`:

1. **Feature importance plot** (RF or model-specific)
   - Bar chart of top-N most important features
   - Saved to `eval_dir/feature_importance.png`

2. **SHAP summary plot** (on test set sample)
   - Global SHAP values
   - Saved to `eval_dir/shap_summary.png`

3. **Feature count report**
   - Total features computed vs features used by model
   - Saved as `eval_dir/feature_report.json`

## Implementation Steps

### Step 1: Update `ExtractFeaturesConfig`
- Replace `individuals: list[str] = INDIVS_SIMBA` with `individuals: list[str]` (required)
- Replace `bodyparts: list[str] = BPTS_SIMBA` with `bodyparts: list[str]` (required)
- Remove `from behavysis.constants import BPTS_SIMBA, INDIVS_SIMBA`

### Step 2: Update `ClassifyBehaviourConfig`
- Add `individuals: list[str]` and `bodyparts: list[str]` fields
- Add `validate_bodypoint_match(extract_cfg: ExtractFeaturesConfig) -> None` method

### Step 3: Rewrite `extract_features.py`
- Remove all SimBA-specific constants (`BP_XY_IDX`, `MOVEMENT_BP_NAMES`, `SIMBA_BODY_PARTS`, `BPMAP_SIMBA`, `INDIVS_SIMBA`)
- Remove SimBA-specific imports from `behavysis.constants.bodypoints`
- New `_pivot_to_wide(keypoints_df, individuals, bodyparts)` → dict of arrays
- New feature group functions with generic naming
- `compute_features()` (was `compute_simba_features`) returns config-driven output

### Step 4: Update `classify_behaviour()` in `classify.py`
- Add validation between ExtractFeaturesConfig and ClassifyBehaviourConfig individuals/bodyparts

### Step 5: Update `BehaviourClassifier` diagnostic reporting
- Add `_report_feature_importance()` method
- Add `_report_shap()` method
- Call from `train()` after evaluation

### Step 6: Update tests
- Rewrite `test_simba_features.py` → generic feature tests
- Add test for config validation mismatch
- Add test for different bodypoint configs (e.g. 1 individual × 4 bodyparts)

### Step 7: Validate
- `uv run ruff check src/`
- `uv run pytest -m "not slow and not integration and not gpu"`

## Success Criteria

- [ ] `ruff check src/` passes
- [ ] `pytest -m "not slow and not integration and not gpu"` passes
- [ ] Feature extraction works with any individuals + bodyparts list
- [ ] Non-SimBA bodypoint config produces valid features (no errors, no NaNs/Infs)
- [ ] Config mismatch between extract and classify raises clear error
- [ ] Diagnostic plots (feature importance, SHAP) saved to eval dir during training
- [ ] All feature columns named programmatically, no SimBA-specific hardcoded names
- [ ] No import of `BPMAP_SIMBA` or `INDIVS_SIMBA` in `extract_features.py` or `experiment_config.py`

## Notes

- `constants/bodypoints.py` remains unchanged — `BPTS_SIMBA`, `BPTS_CENTRE`, etc. are still used by other modules (analyse/freezing, analyse/speed, etc.)
- `BehaviourClassifierConfig` in `config.py` keeps `feature_start_col` and `nfeatures` for backward compat with the trainer code; we add `individuals` and `bodyparts` as new fields
- The `_tail_end_relative_rolled` function is removed — it's purely semantic
- Rolling windows are computed on ALL base numeric features (distances + movements + hull + cdist + totals) — this can produce thousands of columns, which RF handles
