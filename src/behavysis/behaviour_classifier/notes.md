Here's my assessment. Let me be direct.

## The core problem

**The API is optimized for the batch case and treats single-model training as an afterthought.** In practice, you almost always train one model, iterate on it, then maybe train a few more. This API makes that painful.

## Specific issues

### 1. No clean single-model entry point

To train one model you have to write:

```python
initial_train(contract_fp, "rf", MODEL_REGISTRY["rf"])
```

This requires knowing about `MODEL_REGISTRY`'s internal structure and the `Callable[[Path], BaseAdapter]` factory signature. There should be:

```python
train_model(contract_fp, "rf")
```

### 2. `train_all_models` is a blunt instrument

- Currently only trains `["xgb_v2"]` anyway, so it's effectively a single-model call with a misleading name
- If `ROUTINE_MODELS` grows, you can't add one new model without retraining all previous ones from scratch
- No skip-if-exists, no resume, no partial retry — if model 3 of 5 fails you retrain all 5

### 3. The dependency graph is inverted

```
train_all_models → initial_train → train
```

The batch function calls the single-model function. It should be the reverse: `train_model` is the primitive, `train_all_models` is a thin loop around it. Right now the single-model path (`initial_train`) feels like internal plumbing, not a first-class API.

### 4. Fragile path navigation

`train()` does `ClassifierFp(config_fp.parent.parent.parent)` — that `parent.parent.parent` is a silent contract with the directory layout in `storage.py`. If the structure ever changes, this breaks with no clear error. `ClassifierFp` should have a `from_config_fp()` factory or accept a config path directly.

### 5. Save-then-immediately-load in `train()` (lines 110-114)

The adapter is saved, then immediately loaded. If this is a validation step, name it. If not, it's wasted I/O on every training run. Either way, the intent is opaque.

### 6. `__init__.py` has a real bug

`"train"` is in `__all__` but never imported. `initial_train` is imported but not in `__all__`. If someone does `from behavysis.behaviour_classifier import train` they'll get an `ImportError`.

### 7. No model discovery

There's no way to ask "what models exist?" or "what's the latest iteration of rf?" without `ls classifiers/`. No `list_models()`, no `latest_iteration()`. The `_next_iteration` function is private and only tells you the _next_ number, not the current state.

### 8. Template has a typo

`train_behaviour_classifier.py:231` references `"gxb_v2"` — should be `"xgb_v2"`.

## What a clean API would look like

```python
# The primitive — does one thing
train_model(contract_fp, "rf")                          # next iteration
train_model(contract_fp, "rf", iteration=3)             # retry iteration 3?
                                                         # (needs overwrite policy)
# The batch — a thin loop
train_all_models(contract_fp)                           # trains ROUTINE_MODELS
train_all_models(contract_fp, models=["rf", "xgb_v2"])  # explicit subset

# Discovery
list_models(contract_fp)            # → {"rf": [1, 2, 3], "xgb_v2": [1]}
latest_iteration(contract_fp, "rf") # → 3
```

The key insight: `train_model(contract_fp, name)` should be the **public, documented, obvious** entry point. `train_all_models` should be a five-line wrapper around it. Right now the relationship is backwards.

Want me to put together a concrete refactoring plan?
