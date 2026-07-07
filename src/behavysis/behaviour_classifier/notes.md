I read the classifier module and everything it touches on-disk (`storage.py`, `config.py`, `adapter.py`, `data.py`, `evaluation.py`, `registry.py`), plus the actual pipeline consumer (`funcs/behaviour/classify.py`, `experiment.py`), the feature writer, and the scaffolding template. I did not read the README. I validated the riskier claims by running Python (pydantic behaviour) and grepping. Here's the verdict.

## Verdict

The **on-disk artifact tree is sound** and `train()` produces exactly what `storage.py` documents. But the surrounding wiring has real defects: a scaffolding template that is pure fiction, a silently-broken override path, a train/inference feature skew that makes inference crash, and systemically broken logging. Harsh but fair: the directory structure is fine; the code that fills and consumes it is not.

## What is correct

`train()` writes the full tree per `storage.py:1-18`:

```
behaviour_models/{behaviour}/production.yaml, leaderboard.yaml
  {model_type}/config.yaml, active.yaml
    versions/{version}/  model.joblib | model.pt+scaler.joblib, metadata.yaml, dataset_manifest.yaml, evaluation/
```

Version numbering (`_next_version`, `v{seq:03d}_{ts}`), the three-way split index remapping (`_three_way_split:719-731`), and auto-promotion logic are all internally consistent and correct.

## Critical

1. **Train/inference feature skew — inference cannot run.** Training drops `frame`: `data.py:25` does `.set_index("frame").to_numpy()`, so X = features only. The features parquet is written _with_ a `frame` column (`extract_features.py:657`). The pipeline caller passes it as a column: `classify.py:47` → `predict(features_df.to_pandas())`, and `predict` does `x = features_df.to_numpy()` (`behaviour_classifier.py:569`) with no frame drop. Result: inference X has `n_features+1` columns → `scaler.transform` raises a feature-count mismatch. Separately, `predict` returns `index=features_df.index` (a positional RangeIndex, not real frame ids), so `classify.py:51` records fake frames even if shapes matched.

2. **`train_all_models` overrides are silently discarded.** `behaviour_classifier.py:241` filters with `hasattr(TrainingRecipe, k)`. On pydantic v2, model fields are **not** class attributes — I verified `hasattr(TrainingRecipe, 'seed') == False`. So _every_ override (`individuals`, `bodyparts`, …) is dropped and configs are written with defaults. Silent no-op.

3. **Scaffolding template `templates/train_behav_model.py` is entirely non-functional.** It imports/calls an API that does not exist: `BehaviourClassifierConfig` (only `TrainingRecipe` exists), `boris2behaviour` (actual name `boris_to_behaviour`), methods `.create_all_from_project_dir` / `.create_from_project` / `.create` / `clf.train()` (none exist), `train_all_models(..., config_overrides=...)` (signature is `**overrides`; would be dropped anyway per #2), and dir `7_scored_behaviour` (constant is `7_behaviour_scored`). It fails at import. If this is the "generated project" a user gets, it is 100% broken.

## High

4. **Broken logging (systemic).** loguru only does `{}`-formatting, yet these use `%s`, so they log literal `%s`: `behaviour_classifier.py:272,290,400,426`; `adapter.py:102,168,186`; `evaluation.py:243,279`; `classify.py:66`. The module is internally inconsistent — `:114` and `:217` correctly use `{}`. (Bonus: `:218` logs `v{}` on an already-`v…`-prefixed version → `vv001_…`.)

## Medium

5. **Existence guards check the directory, not the model.** `model_fp()` returns the _version dir_ (`storage.py:91-98`), which `train()` creates at step 5 before saving. `promote`/`promote_to_production`/`_load_version` guard on `model_fp(...).exists()` — a version whose save failed still passes, then `_load_adapter` fails later with a murky error. `model_fp` is also misnamed (returns a dir).
6. **`_model_types` hardcodes the root.** `behaviour_classifier.py:655` uses `"behaviour_models"` literally instead of `storage.MODEL_ROOT` / `behaviour_dir()`. Diverges silently if the root ever changes.
7. **Inference output doesn't match the project schema.** `predict` returns a legacy pandas MultiIndex (`behaviour_classifier.py:575-584`), not the canonical polars `BEHAVIOUR_PREDICTED_SCHEMA (frame, behaviour, prob, pred)`. The consumer reshapes manually and **discards** predict's `pred`, recomputing it with a different pcutoff (`classify.py:53-55`) — so predict's `pred` (`:573`) is dead in the pipeline.
8. **Column level-name mismatch.** `predict` uses `[BEHAVIOUR, OUTCOME]` = `"outcome"` (`:578`); `save_evaluation_results` hardcodes `"outcomes"` plural (`evaluation.py:178`).

## Low

9. **Dead code:** `training_data_dir` (`storage.py:42`) is never used and isn't in the documented tree.
10. **Val leakage:** test is video-grouped (`StratifiedGroupKFold`), but train/val is a plain stratified split (`_three_way_split:712`) ignoring video groups → val frames leak from train's videos, making val metrics optimistic. Likely intentional (val = reporting/early-stop) but undocumented.
11. **Torch double-val:** the top-level `val_idx` (reported as `n_val`) is a different set from the internal val the torch model actually early-stops on (`adapter.py:145`), so `n_val` is misleading for torch.
12. `TrainingRecipe.individuals/bodyparts` are stored but unused by training (filtering happens only in `classify.py`).

## Suggested remediation order (not yet executed)

1. Fix #1: drop `frame` in `predict` (or accept frame-indexed input) and carry real frame ids through — this is what actually breaks the pipeline.
2. Fix #2: filter overrides against `TrainingRecipe.model_fields` instead of `hasattr`.
3. Rewrite/regenerate the template to the real API (#3).
4. Convert all `%s` logging to `{}` (#4).
5. Address the rest (guards, hardcoded root, schema-conformant `predict`) as cleanup.

Want me to implement these? If so, I'd start with #1–#2 (behavioural bugs) plus a reproducing test, then #3–#4. Tell me if "generated project structure" meant the on-disk artifact tree, the scaffolding template, or both — it changes where I focus.
