Now I have the full picture. Here's a thorough answer to both questions.

---

## 1. EXPERIMENT / BEHAVIOUR splitting: training is broken, inference is correct

### Inference path (`classify_single` → `predict`)

The features DataFrame has `[FRAME, feat1, ...]` — **no EXPERIMENT**.

`SklearnAdapter.predict()` checks `if EXPERIMENT in df.columns` → False → `experiment = None`.

`_predict_postprocess(prob, frame, experiment=None)` → `if None:` → False → EXPERIMENT not added.

`label_bouts` sees no EXPERIMENT → `group_cols = [BEHAVIOUR]`. BEHAVIOUR was hardcoded in `_predict_postprocess` as `recipe.behaviour_name`. BOUT_IDs computed per-behaviour. **Correct.**

**Verdict: inference is fine. No bout-id mixing.**

### Training path (`train_model`) — two bugs

**Bug A: `label_bouts` references a column that doesn't exist**

`train_model:120`:

```python
df = load_all_data(...)   # → [FRAME, ACTUAL, feat1, ..., EXPERIMENT]
df = label_bouts(df, ACTUAL)
```

`load_all_data` (data.py:76-80) **drops** BEHAVIOUR after filtering:

```python
.filter(BEHAVIOUR == behaviour_name)
.select(FRAME, pl.col(ACTUAL).replace(...))   # ← BEHAVIOUR gone
```

Then `label_bouts` _unconditionally_ appends BEHAVIOUR to group_cols:

```python
group_cols.append(BEHAVIOUR)   # always, even if column missing
```

The `pl.col(BEHAVIOUR).ne_missing(pl.col(BEHAVIOUR).shift())` lazy expression resolves at `.with_columns()` time → **ColumnNotFoundError**. This is a dead code path that has likely never been exercised. The training df only has one behaviour anyway, so BEHAVIOUR isn't needed for grouping — but the code references it unconditionally.

**Bug B: `_predict_postprocess` crashes on Series truthiness** (the #1 bug from my earlier review). Hit whenever EXPERIMENT is present in the prediction input — which happens during training eval (`train_model:142-143`).

**Verdict: the training path has two blocking bugs.** You've probably only tested inference successfully, which is why the classifier "works" in practice.

---

## 2. Bout conversion code: yes, overcomplicated

Here's a direct comparison of what exists vs what your proposal would simplify:

### Current architecture (wide + dynamic columns)

```
BoutStruct model  ←  consumer of get_bouts_struct()
       ↑
get_bouts_struct()  ←  scans columns at runtime to figure out which are sub_behaviours
       ↑
predicted_to_scored()  ←  pre-populates ALL sub_behaviour columns with TRUE_NEG
       ↑
frames2bouts()  ←  iterates dynamic columns, builds sub_behaviour dicts per bout
       ↑
Bouts model {start, stop, dur, behaviour, actual, sub_behaviour: dict}
       ↑
bouts2frames()  ←  needs BoutStruct to know which columns to reconstruct
```

The `BoutStruct` / `Bouts` / `Bout` pydantic models exist solely to carry dynamic column metadata through the pipeline. `get_bouts_struct` runtime-scans the DataFrame to discover what columns exist. `predicted_to_scored` materializes empty columns for all possible sub_behaviours across all behaviours, even though each behaviour only needs its own. `frames2bouts` then scans them again.

### Your proposal: `[frame, behaviour, sub_behaviour, value]`

This mirrors `ANALYSIS_SCHEMA` (`[frame, individual, measure, value]`) — the project's established long-form pattern:

```python
BEHAVIOUR_SCORED_SCHEMA: SchemaDict = {
    FRAME:         pl.Int64,
    BEHAVIOUR:     pl.Utf8,      # e.g. "attack", "mount"
    SUB_BEHAVIOUR: pl.Utf8,      # e.g. "actual", "pred", "intensity", "confidence"
    VALUE:         pl.Int64,     # TRUE_POS/TRUE_NEG/FALSE_POS/UNSURE for categorical
}
```

Then:

- `predicted_to_scored` becomes: unpivot PRED into rows with `sub_behaviour="pred"`, add `sub_behaviour="actual"` rows (all UNSURE).
- `frames2bouts` becomes: filter `sub_behaviour=="actual" & value==TRUE_POS` → `vect2bouts` per behaviour → join in other sub_behaviour values via `group_by(bout_range, sub_behaviour).agg(mode)`.
- `bouts2frames` becomes: explode bout ranges → cross join with sub_behaviour values.
- No more `get_bouts_struct`, no more `BoutStruct` model, no dynamic columns.

The schema is _fixed_. Adding a new sub_behaviour is adding data rows, not schema columns.

**Tradeoff**: ~3-5x row count (one row per sub_behaviour per frame per behaviour instead of one per frame per behaviour). With typical values (3 behaviours × 3 sub_behaviours = 9 rows per frame), this is negligible. Polars handles this efficiently, and it's the same pattern the rest of the codebase uses for `ANALYSIS_SCHEMA`.

It's the right move — consistent with project conventions and eliminates the dynamic-column complexity entirely.
