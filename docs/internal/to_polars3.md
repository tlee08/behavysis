# Polars migration strategy & schema redesign

**Session ID:** ses_0e7232d2effeaX89m1cB215MpP
**Created:** 7/1/2026, 12:09:12 AM
**Updated:** 7/1/2026, 12:09:59 AM

---

## User

Read @docs/internal/to_polars2.md and @docs/internal/to_polars.md .
Give your recommendations on changing to polars. How are we going to handle the different format of polars (using long instead of our current wide approach)?

Be a principle data engineer and data scientist. Be harsh and honest. Be elegant and foresightful with planning data structures that fit together beautifully, simply, elegantly, and streamlined.
Use karpathy guidelines, context7, and tavily to inform.

---

## Harsh Assessment

Your `to_polars2.md` analysis is sound but incomplete. Here's what you're missing:

### 1. Long vs Wide: You're Asking the Wrong Question

The question isn't "long vs wide." The question is: **what is each DataFrame semantically?**

| Type      | What it IS                                    | Correct Form                          |
| --------- | --------------------------------------------- | ------------------------------------- |
| Keypoints | Observations of (indiv, bpt, coord) per frame | **Long** - each row = one measurement |
| Behaviour | Observations of (behav, outcome) per frame    | **Long** - each row = one label       |
| Features  | Input vectors for ML                          | **Wide** - each row = one sample      |
| Analysis  | Aggregated measurements                       | **Wide-ish** - depends on consumer    |

Your current wide form conflates **schema with data**. `("nose", "x")` is not a column name - it's data. The MultiIndex is papering over a fundamental modeling error.

### 2. The aggregation loops are embarrassing

`analysis_agg_df.py` lines 34-60 and 63-91: looping column-by-column, constructing Series, concatenating. This is **anti-Pattern Polars 101**.

```python
# Your current approach (embarrassing)
for i, col in enumerate(analysis_df.columns):
    vect = analysis_df[col]
    summary_df_ls[i] = pd.Series({...})

# What Polars makes trivial (single expression)
df.group_by("measures").agg([
    pl.col("value").mean().alias("mean"),
    pl.col("value").std().alias("std"),
    pl.col("value").quantile(0.25).alias("Q1"),
    # ...
])
```

### 3. The `DFMixin` class hierarchy deletes itself

If you go long-form:

- No MultiIndex to validate
- Schema = dict of column types
- Read/write = `pl.read_parquet()` / `df.write_parquet()`
- No subclass needed

The entire `df_classes/` package collapses to ~5 schema definitions + utility functions.

### 4. Pivot-on-demand is correct, but you understate it

Your Paradigm B (wide-ish, pivot-on-demand) is the right answer. You should:

1. **Store long** (parquet compression loves it)
2. **Pivot at consumption** (ML features, plotting)
3. **Never persist wide** (unless ML model artifacts)

The cost of pivoting is amortized against computation. A single `df.pivot()` before sklearn is free compared to the model training.

### 5. The Bout framing problem

`frames2bouts` / `bouts2frames` becomes a **single expression** in long form:

```python
# frames2bouts in long form
df.filter(pl.col("pred") == 1).group_by("behav").agg([
    ((pl.col("frame") - pl.col("frame").shift(1)) != 1).alias("is_new_bout"),
    # then cumulative sum to get bout IDs, then group again
])
```

Actually simpler: use `pl.col("pred").diff()` and find transitions. The current implementation (lines 125-135 in behaviour_df.py) with numpy concatenation is fragile and index-dependent.

### 6. What you're not discussing: Lazy evaluation

Polars LazyFrames change the **entire pipeline architecture**:

```
Current:  Read → Validate → Transform → Validate → Write → Repeat
Lazy:     Read → Filter → Transform → Aggregate → Collect → Write (once)
```

Your current pipeline has 352 lines of `test_df_mixin.py` testing validation behavior that **vanishes** with lazy evaluation. You validate at the end when you `collect()`, not at every step.

### 7. Data size matters

Your doc says "1.3M rows... small for Polars." This is correct. But it misses the point:

- 1 video = 1.3M rows × 12 bpts × 2 indivs = 31M rows in long form
- 50 experiments × 10 videos = 15.5B rows

**At this scale, LazyFrames + streaming are not optional.** You need to think about:

- Predicate pushdown (filter before reading)
- Projection pushdown (select columns before reading)
- Streaming execution (don't materialize full dataset)

### My Recommendation

1. **Accept Paradigm A + B** from your doc (long storage, pivot on demand)
2. **Delete the class hierarchy** - replace with schema dicts + free functions
3. **Start with `analysis_agg_df.py`** - highest payoff, lowest risk (no downstream consumers, pure computation)
4. **Then `behaviour_df.py`** - `frames2bouts`/`bouts2frames` become cleaner
5. **Then `keypoints_df.py`** - highest volume, biggest compression win
6. **Last: GUI viewer** - needs boundary conversion (`df.to_pandas()` at view layer)

The GUI viewer should **never** drive data architecture. It's a visualization concern. Convert at the boundary.

### What you should stop doing

- Stop calling `clean_and_validate` repeatedly. Validate at write boundaries only.
- Stop using Enums as namespace containers for strings. Just use strings with type hints.
- Stop treating MultiIndex as a schema. It's not.
- Stop looping over columns. Learn Polars expressions.

---

Do you want me to draft the concrete schema definitions for each DataFrame type, or the migration order with specific file-by-file changes?

---
