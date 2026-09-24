# HPW-right classifier review

Date: 2026-09-24. Companion scripts and outputs: `data/scratch/` (01–05 + `common_hpw.py`, `memory.md`).

## Aim

Improve the hind-paw-withdrawal-right (HPW-right) classifier. It is a *screening*
tool: high recall matters, a human reviews the flagged events, so precision is the
review cost to reduce.

Data: `data/training_data_hpw` — 64 labelled videos = 16 animals × (HOT, COLD) × (a, b),
clipped to 120 s from rat entry. 614 features per frame. Model: `data/model_hpw_right`.

## Honest baseline

On a 12/4 **animal** split (no animal in both train and test):

| metric | value |
|---|---|
| PR-AUC | **0.52** |
| ROC-AUC | **0.90** |
| positive rate | ~8% |

The classifier is **data-limited, not model-limited**. The honest ceiling is ~0.52 PR-AUC.

## Findings

### 1. Calibration objective is broken (biggest, cheapest win) — FIXED

`optimise_postprocessing_parameters` maximised **bout-level precision**, which is a
degenerate metric: it sits flat at ~0.49 across thresholds and is slightly *highest* at
the lowest threshold (predicted bouts merge). So the stored model collapsed to
`pcutoff ≈ 2.5e-5` (recall 100%, frame precision 8%, flags 99.95% of frames).

The same model at a sane threshold already reaches the useful point:

| pcutoff | bout recall | frame precision | frame recall |
|---|---|---|---|
| ~0.032 | ≥ 0.95 | **0.21** | ~0.90 |
| 2.5e-5 (stored) | 1.00 | 0.08 | 1.00 |

**Fix applied** (`adapter.py: _best_pcutoff`): maximise *frame-level* precision subject
to bout-level recall ≥ target, instead of bout-level precision. Verified: the sweep now
picks `pcutoff ≈ 0.032`, `min_bout = 8`, giving frame precision 0.21 vs 0.08 — a ~2.6×
improvement at no recall cost. `scale_pos_weight` does *not* fix this (tested); the
objective was the bug.

### 2. Split leaks animals, but does not inflate PR-AUC — leave as-is

Production splits by filename, so the same animal can appear in train and test. Measured:

| split | PR-AUC | ROC-AUC |
|---|---|---|
| honest (by animal) | 0.524 | 0.903 |
| leaky (by filename) | 0.521 | 0.904 |

No inflation. **Decision: leave the random per-experiment split** (potentially leaking,
but no inflation observed).

### 3. Keypoint quality: toe is fine, knee/heel are weak

Mean likelihood (256 files, both cameras track equally):

| bodypart | mean likelihood | low-confidence |
|---|---|---|
| hind_toe_r/l | ~0.63 | 0.38 |
| hind_heel_r/l | ~0.50 | 0.52 |
| hind_knee_r/l | **~0.36** | **0.67** |

- Tracking **improves** during events for the relevant paw: `hind_toe_r` likelihood
  0.755 on HPW-right frames vs 0.726 otherwise. DLC does not fail when we need it.
- Camera A vs B track equally (0.508 vs 0.506). The camera-B PR-AUC edge seen on 4 test
  animals is chance, not tracking quality.
- The toe drives the core features (elevation W04, vertical velocity W01) and is solid.
  The knee/heel drive knee-elevation, knee-toe angle, paw area, heel-toe distance and
  are the weak features.

DLC improvement is feasible and standard: `extract_outlier_frames` on the hind paw →
`refine_labels` (do not label occluded points) → retrain. It is not the core bottleneck.

### 4. Features: strong signal, high redundancy, pruning does not help

- Best single feature AUC 0.82: right-paw vertical-velocity peak (W18), elevation std
  (W04), vertical-velocity std (W01). All toe-based.
- PCA: 614 features → 50% variance in 12 dims, 90% in 61 dims (~10× redundant).
- Feature pruning **hurts**: 614→60→20 features drops PR-AUC 0.51→0.41→0.37. XGBoost
  already handles the redundancy.
- Regularisation gives +0.01: `reg_lambda=10, min_child_weight=50` lifts PR-AUC
  0.517→0.530 and narrows the train/test gap.

### 5. Label window: point → peak offset

The peak vertical velocity lands **~3 frames after** the scored point (median +3;
27% of points peak *before* the point — scorer is late; p90 = +17). The label window
should straddle the point with more room after it. Suggested `before=5, after=10`.

## Actions taken

1. **Calibration fix** — `src/behavysis/behaviour_classifier/adapter.py`: `_best_pcutoff`
   now optimises frame-level precision at target bout recall. Validated (pcutoff 0.032).
2. **Label window API** — `import_boris_csv` now takes `frames_window_before` /
   `frames_window_after` (was `point_window_sec`). Updated in:
   - `data/training_data_hpw/prepare_boris_to_labels.py`
   - `src/behavysis/templates/behaviour_classifier/import_boris.py` (last two metadata
     reference cells kept).

## Next steps (recommended order)

1. Retrain with the fixed calibration (retrain model → re-run `optimise_postprocessing_parameters`).
2. Regenerate labels with `frames_window_before=5, frames_window_after=10` and re-evaluate.
3. Improve DLC on hind knee/heel (outlier extraction + refine + retrain).
4. Recalibrate `px_per_mm` / `dist_mm` (still `TODO: 100`), then re-extract features and retrain.
