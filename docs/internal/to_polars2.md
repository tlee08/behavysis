How can I change over to using polars? It would be a big change considering I use index, and I use multiindex columns?
What should be my paragigms? This is actually a chance to design my data better. Be a principle data scientist and sketch out some options and considerations.

---

# Polars Migration: Paradigm Sketch

## The fundamental tension

pandas MultiIndex gives you:

- **Schema as axes** — column names ARE your data model (`df[(behav, PRED)]`)
- **Computation as axis traversal** — `loc[idx[:, :, bpt, X]]`, `get_level_values`, `columns.unique(level=)`

Polars has none of that. Rows are anonymous, columns are flat strings. Your schema must become data. Your axis traversal must become filters.

**This is good.** The current code has 48+ `df[(tuple)]` accesses, 18 `columns.unique(level=)` calls, 11 `get_level_values` — every one is a latent design smell that Polars would make explicit.

---

## Usage census (what you'd touch)

| Pattern                           | Count | Polars replacement                                       |
| --------------------------------- | ----- | -------------------------------------------------------- |
| `df[(behav, PRED)]`               | 48    | `df.filter(behav=...).select("pred")` or `df.pivot(...)` |
| `columns.unique(level=)`          | 18    | `df.select("behavs").unique()`                           |
| `get_level_values(FRAME)`         | 11    | `df.select("frame")` — it's a column now                 |
| `pd.IndexSlice` / `loc[idx[...]]` | 19    | `df.filter(...)`                                         |
| `unstack/stack/reorder_levels`    | 4     | `df.pivot(...)` / `df.melt(...)`                         |
| `MultiIndex.from_frame`           | 6     | gone entirely                                            |

---

## Three paradigms (with real-data sketches)

### Paradigm A: **Long-form (tidy data)** — the Polars-native answer

Every dimension becomes a column. Every measurement goes vertical.

**Keypoints before:**

```
index: frame=0..N
columns: (scorer, individual, bodypart, coord) → 4-level MultiIndex
         ("DLC", "single", "nose", "x") → value
         ("DLC", "single", "nose", "y") → value
```

**Keypoints after:**

```python
shape: (N_frames × N_indivs × N_bodyparts, 6)
┌───────┬────────┬────────────┬──────────┬───────┬───────┬─────────────┐
│ frame ┆ scorer ┆ individual ┆ bodypart ┆ x     ┆ y     ┆ likelihood  │
│ i64   ┆ str    ┆ str        ┆ str      ┆ f64   ┆ f64   ┆ f64         │
╞═══════╪════════╪════════════╪══════════╪═══════╪═══════╪═════════════╡
│ 0     ┆ DLC    ┆ mouse1     ┆ nose     ┆ 123.4 ┆ 56.7  ┆ 0.99        │
│ 0     ┆ DLC    ┆ mouse1     ┆ tailbase ┆ 200.1 ┆ 89.3  ┆ 0.98        │
│ 1     ┆ DLC    ┆ mouse1     ┆ nose     ┆ 124.1 ┆ 57.0  ┆ 0.99        │
│ 1     ┆ DLC    ┆ mouse1     ┆ tailbase ┆ 199.8 ┆ 88.9  ┆ 0.97        │
│ ...   ┆ ...    ┆ ...        ┆ ...      ┆ ...   ┆ ...   ┆ ...         │
└───────┴────────┴────────────┴──────────┴───────┴───────┴─────────────┘
```

Row count: `N_frames × N_bodyparts × N_individuals`. Same data density as before (wide form had `N_frames` rows with `N_bodyparts × 3` columns — this just reshapes).

**All those operations become trivially readable:**

```python
# Old: df.columns.get_level_values("individuals").unique()
# New:
df.select(pl.col("individual")).unique()

# Old: filter_mask = ~df.columns.get_level_values(INDIVIDUALS).isin([PROCESSED, SINGLE])
# New:
df.filter(~pl.col("individual").is_in(["processed", "single"]))

# Old: df.loc[:, idx[:, :, :, X]] *= width_scale
# New:
df.with_columns(
    pl.when(pl.col("coord") == "x")
    .then(pl.col("value") * width_scale)
    .otherwise(pl.col("value"))
)
```

Wait — the above example with `coord == "x"` assumes a "value" column. But Paradigm A has `x`, `y`, `likelihood` as separate columns already. That's better:

```python
df.with_columns(pl.col("x") * width_scale, pl.col("y") * height_scale)
```

**BeahvDf before:**

```
index: frame
columns: (behaviour, outcomes) → ("grooming", "pred"), ("grooming", "actual"), ...
```

**BeahvDf after:**

```python
shape: (N_frames × N_behaviour, 5)
┌───────┬──────────┬──────┬──────┬────────┐
│ frame ┆ behav    ┆ prob ┆ pred ┆ actual │
│ i64   ┆ str      ┆ f64  ┆ i64  ┆ i64    │
╞═══════╪══════════╪══════╪══════╪════════╡
│ 0     ┆ grooming ┆ 0.95 ┆ 1    ┆ 1      │
│ 0     ┆ rearing  ┆ 0.03 ┆ 0    ┆ 0      │
│ 1     ┆ grooming ┆ 0.91 ┆ 1    ┆ 1      │
│ ...   ┆ ...      ┆ ...  ┆ ...  ┆ ...    │
└───────┴──────────┴──────┴──────┴────────┘
```

```python
# Old: df[(behav, PRED)] == TRUE_POS
# New: df.filter(pl.col("behav") == "grooming", pl.col("pred") == 1)

# Old: vect2bouts_df(vect == 1)  # vect is a slice of wide df
# New: vect2bouts_df(df.filter(pl.col("pred") == 1).select("frame"))
```

✅ **Net win**: every `.get_level_values(...)`, `.columns.unique(...)`, tuple-column-access becomes a simple `.filter()`, `.select()`, `.unique()`. More readable, more composable.

---

### Paradigm B: **Wide-ish (pivot-on-demand)** — closer to current code

Store long, pivot to wide for computation. Example: keypoints stored as Paradigm A above, but pivoted to wide for ML feature extraction:

```python
# Long → wide for ML
wide = df.pivot(
    index="frame",
    on=["individual", "bodypart"],
    values=["x", "y", "likelihood"]
)
# Columns: frame, mouse1_nose_x, mouse1_nose_y, mouse1_nose_likelihood, ...
```

Then feed to sklearn/xgboost. This is what you'd do anyway for ML features.

✅ **Pragmatic compromise**: long for pipeline, wide for ML. Polars can handle the pivot efficiently.

---

### Paradigm C: **Struct columns** — keep hierarchy explicit

```python
# Keypoints as struct
df = pl.DataFrame({
    "frame": [0, 0, 1, 1],
    "individual": ["mouse1", "mouse1", "mouse1", "mouse1"],
    "bodypart": ["nose", "tailbase", "nose", "tailbase"],
    "coords": [
        {"x": 123.4, "y": 56.7, "likelihood": 0.99},
        {"x": 200.1, "y": 89.3, "likelihood": 0.98},
        {"x": 124.1, "y": 57.0, "likelihood": 0.99},
        {"x": 199.8, "y": 88.9, "likelihood": 0.97},
    ]
})

# Access struct field
df.select(pl.col("coords").struct.field("x"))
# Scale x
df.with_columns(
    (pl.col("coords").struct.field("x") * width_scale).alias("x_scaled")
)
```

❌ **Not recommended.** Struct dot notation is clunky in expressions. Unnest/flatten adds ceremony. The current code accesses individual coordinate values constantly — struct just adds another `.struct.field("x")` layer. Long form is cleaner.

---

## Recommendation: Paradigm A with tactical B

| Data type           | Storage form                                               | Why                                                                                                |
| ------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `KeypointsDf`       | Long (frame, scorer, indiv, bpt, x, y, likelihood)         | Natural fit. 48 tuple accesses collapse to filters.                                                |
| `BehavDf`           | Long (frame, behav, prob, pred, actual)                    | Behaviours are observations, not columns. `bouts2frames`/`frames2bouts` become groupby operations. |
| `AnalysisDf`        | Long (frame, indiv, measure, value)                        | Same as keypoints — measurements per individual are observations.                                  |
| `FeaturesDf`        | Wide-ish (frame, feature_1, feature_2, ...)                | Features ARE columns. ML models expect this shape. Keep wide.                                      |
| `AnalysisSummaryDf` | Long (indiv, measure, agg, value) or wide with agg columns | Aggregations are naturally columnar.                                                               |
| `AnalysisBinnedDf`  | Long (bin_sec, indiv, measure, agg, value)                 | Similar. Pivot for plotting.                                                                       |

---

## What goes away entirely

1. **`DFMixin.clean_and_validate`** — no index/column names to enforce. Schema becomes a Polars schema dict: `{"frame": pl.Int64, "x": pl.Float64, ...}`
2. **`DFMixin` class hierarchy** — `read`/`write` are `pl.read_parquet(fp)` and `df.write_parquet(fp)`. No subclass needed.
3. **`_enum_values`** — already dead.
4. **`pd.IndexSlice`** — 19 occurrences replaced by `.filter()`.
5. **Sort validation** — no index to sort. Sort columns explicitly when needed.
6. **`tuple` column access** — 48 occurrences gone. Just filter a string column.

---

## Hard things that get harder (or different)

1. **`bouts2frames` / `frames2bouts`** — these iterate nested MultiIndex column groups (`df[behav]` returns a sub-dataframe). In long form, this becomes `df.filter(pl.col("behav") == "grooming")`. Different mental model, same result. The `vect2bouts_df` function stays as-is since it operates on a 1D series.

2. **`predicted2scored`** — currently copies column groups between formats. In long form: filter predicted rows, transform, assign `actual` column. Simpler.

3. **Pivot cost for wide-form consumers** — if downstream code expects wide keypoints (e.g., `analyse.py` accessing multiple bodyparts simultaneously), you'd pivot long→wide on demand. Polars pivot is fast, but it's a step that didn't exist before.

4. **File size** — long form stores dimension values repeated per row. `frame` repeats `N_bodyparts × N_indivs` times. Parquet columnar compression handles this well (run-length encoding), but worth benchmarking.

---

## Migration strategy: river, not flood

```
Phase 1: Adapter layer
  Keep pandas internals. Add polars read/write alongside.
  Convert at edges: polars → pandas at IO boundaries.

Phase 2: Flip one module
  Pick behav_df (simplest schema: 2 level names).
  Rewrite BehavScoredDf.bouts2frames/frames2bouts etc in long-form polars.
  Verify with existing tests.

Phase 3: Flip keypoints
  Highest volume, most tuple access. Biggest win.
  Rewrite KeypointsDf and all consumers.

Phase 4: Flip analysis chain
  AnalysisDf → AnalysisSummaryDf → AnalysisBinnedDf.
  Already mostly long-form compatible.

Phase 5: Remove pandas
  Kill df_mixin.py, MultiIndex validators, pd imports.
  Delete entire df_classes/ package — replaced by schema definitions + read/write functions.
```

---

## Open questions before starting

1. **How big is your data?** A 30-min video at 30fps with 2 individuals and 12 bodyparts = 30×60×30×2×12 = 1.3M rows in long form. That's small for Polars. But if you have 10 videos per experiment × 50 experiments, worth benchmarking.

2. **Do you need pandas interop?** If DLC outputs pandas, and downstream tools (SimBA, BORIS) expect pandas, you may need a long hybrid period. Measuring the real cost of `pl.from_pandas()` / `df.to_pandas()` at each boundary.

3. **Test coverage?** `test_df_mixin.py` tests pandas MultiIndex behaviour directly. All 352 lines become obsolete. Do you have integration/functional tests that verify pipeline output correctness regardless of internal data representation?

4. **GUI viewer?** `viewer/windows/main.py` accesses `BehavScoredDf.frames2bouts`, `bouts2frames`. The viewer would need to work with polars DataFrames or convert at the boundary.

5. **Lazy vs eager?** Polars LazyFrames enable query optimization — if your pipeline reads parquet → filters → transforms → aggregates → writes, you'd chain all of that without materializing intermediate DataFrames. That's a bigger paradigm shift than just the schema.

---

## Bottom line

Polars is the right call. Your MultiIndex is doing two jobs: schema definition and computation DSL. Polars forces you to make schema explicit (columns) and computation explicit (expressions). The result is less code, fewer abstractions, and a data model that actually represents what the data _is_ — not what pandas makes convenient.

Start with `behav_df.py` — simplest schema, highest payoff-to-risk ratio for learning the patterns. Want me to draft the long-form schema for each data type?

---
