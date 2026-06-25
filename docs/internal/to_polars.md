# df_classes design review for polars port

**Session ID:** ses_11219a6c0ffe7pLxfvyvG3OX6V
**Created:** 6/22/2026, 3:55:56 PM
**Updated:** 6/22/2026, 3:56:55 PM

---

## User

Look at @src/behavysis/df_classes/ . I want to eventually port this to polars. Is this setup currently over-engineered and actually burdensome, or does it have some value? What value?
Use context7, tavily, karpathy guidelines, and caveman speak

---

### Value assessment:

**Real value:**

1. Schema enforcement at IO boundaries — catch bad data on read/write
2. Consistent parquet IO — single pattern, one place to change format
3. MultiIndex name standardization — enforces consistent naming (`frame`, `individuals`, `bodyparts`, `coords`, `behavs`, `outcomes` etc.)
4. Prevents NaN in critical DataFrames (keypoints, behaviours)
5. `init_df` pattern removes boilerplate for creating empty DataFrames with correct structure

**Over-engineered aspects:**

1. Many classes are just enums + mixin with zero methods (e.g., `FeaturesDf`, `AnalysisDf`, `AnalysisCombinedDf`, all 4 collated classes) — 8 of 12 classes are "empty"
2. Class-per-schema creates deep hierarchy for what's effectively just schema + validation
3. Enums used as namespace containers for string constants — `AnalysisCN.INDIVIDUALS` instead of just `"individuals"`

### Porting to polars considerations:

**What breaks:**

1. **No pandas MultiIndex in polars** — polars doesn't have multi-level index/columns. The workaround is:
   - Struct columns for column hierarchy (e.g., `behavs` and `outcomes` become struct columns)
   - Regular columns for index (frame number becomes a regular column)
   - Or use nested structs: `col("behav")` → struct with field `outcome` → etc.

2. **Polars doesn't subclass** — can't `class X(polars.DataFrame)`. Must use `register_dataframe_namespace` for custom methods (but these are classmethods, which don't map well to instance methods on the namespace pattern).

3. **Schema validation pattern changes** — polars has `match_to_schema` method, or you can validate with `collect_schema()`.

4. **All the Python-level loops in `analysis_agg_df.py`** (e.g., `agg_quantitative` loops column-by-column) — these work against polars' vectorized expression API.

**What works well in polars:**

1. Lazy evaluation and query optimization
2. Struct columns for hierarchy (better semantic than MultiIndex)
3. Expression API for group_by aggregation (replaces the manual loop in `agg_quantitative`)
4. Type safety is built in

### My assessment:

The pattern **does have value but it is over-engineered for the volume of code**. The core idea — schema-enforced DataFrame I/O — is genuinely useful in a scientific computing pipeline where data integrity matters. But:

1. The class hierarchy is too deep. 8 of 12 classes add almost nothing.
2. Enums provide no real type safety — they just namespace strings.
3. A single `Schema` config or dataclass per DataFrame type would suffice, rather than separate classes.
4. The `clean_and_validate` loop/call is applied repeatedly (sometimes 3x in a single clip) which is wasteful.

**For polars port:**

- The core mixin pattern should be simplified significantly
- Schema validation can use polars' native `match_to_schema` or just a schema dict
- The enum-based class hierarchy can collapse into ~3-4 schema definitions
- Struct columns replace MultiIndex — which is arguably BETTER

Let me now present this analysis in caveman full mode and tie it to karpathy guidelines.

## Assessment

### Value: Yes, but over-engineered. Maybe 30% too heavy.

**Real value (keep):**

- Schema enforcement at IO boundaries (catching bad data on read/write)
- Consistent parquet IO pattern, single place to change format
- MultiIndex name standardization across pipeline
- NaN prevention in critical data (keypoints, behaviours)
- `init_df` removes boilerplate for empty DataFrames with correct structure

**Over-engineered (question):**

- **8 of 12 classes are empty** — just enums + mixin, zero methods. `FeaturesDf`, `AnalysisDf`, `AnalysisCombinedDf`, all 4 collated classes. They exist solely as schema labels.
- **Class-per-schema creates deep hierarchy** for what's effectively a schema config dict
- **Enums used as string namespace** — `AnalysisCN.INDIVIDUALS` vs `"individuals"`. No actual type safety gain over a plain constant
- **Repeated `clean_and_validate` calls** — sometimes 3x in a single operation chain, wasted work

### Polars port — what breaks:

| Pandas Pattern                                               | Polars Equivalent                                                                 |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| `class X(DFMixin)` — subclassing                             | **Can't subclass** `pl.DataFrame`. Use `register_dataframe_namespace` for methods |
| MultiIndex columns (`behavs`, `outcomes` levels)             | **Struct columns** — `struct` + `unnest`. Arguably better                         |
| MultiIndex index (`frame`)                                   | Regular column (polars has no multi-row-index concept)                            |
| Python loops over columns (see `agg_quantitative` ~15x loop) | Expression API — vectorized, parallel, faster                                     |
| `clean_and_validate` as separate step                        | `DataFrame.match_to_schema()` or check `collect_schema()` inline                  |

### Karpathy lens:

> **Simplicity first.** 8 of 12 classes add nothing but a schema tag. A single `SchemaConfig` dataclass or dict per DataFrame type would suffice. The class hierarchy says "we might add methods later" — but almost none ever got added.

> **Surgical changes.** Porting to polars means all the MultiIndex-based access patterns (`df.columns.unique("bodyparts")`) must change anyway. This is the right time to collapse the hierarchy.

### Recommendation:

| Keep                            | Drop / Flatten                                                     |
| ------------------------------- | ------------------------------------------------------------------ |
| Schema validation on read/write | Class-per-schema inheritance tree                                  |
| Parquet-as-default IO           | Enums as string namespace (use `StrEnum` or plain constants)       |
| `init_df` factory               | `clean_and_validate` called everywhere (validate once at boundary) |
| NaN check                       | Repeated validation in hot paths                                   |

**For polars**: Collapse to a single `Schema` config + a few utility functions. The struct-column approach for hierarchy is cleaner and more performant than MultiIndex. The expression API replaces all the manual column loops in `analysis_agg_df.py`.

Want me to draft a concrete polars port plan?

---
