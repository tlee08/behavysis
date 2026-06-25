# Simplify overengineered DFMixin and df_classes

**Session ID:** ses_10b9ca5e0ffeBmgq8TsrZGH09k
**Created:** 6/23/2026, 10:10:12 PM
**Updated:** 6/23/2026, 10:11:26 PM

---

## User

Look at @tests/unit/test_df_mixin.py and @src/behavysis/df_classes/ . Be a principle data scientist looking to clean up and streamline the code. Be honest, be harsh, be ambitious.
Have I overengineered things (for example, IN and CN being EnumTypes, when they could just be tuples)? What can I just make simpler?
Use karpathy principles, context7, tavily, and speak like caveman.

---

# 🔥 Code Review: `DFMixin` + `df_classes/`

## Verdict: Overengineered. Class explosion. Abstraction worship.

---

### 1. IN/CN as Enums → Tuple would do same thing. Enum is dead wrapper.

Every. Single. Access. is `.value`:

```python
cls.IN.FRAME.value   # → "frame"
cls.CN.BEHAVS.value  # → "behavs"
```

That's not using enum — that's wrapping a string in ceremony. Enum adds:

- `from enum import Enum` in 8 files
- `_enum_values()` helper (code smell — helper exists to undo your abstraction choice)
- `EnumType` import + type annotation in df_mixin.py
- `.value` suffix on every access

Same functionality with tuple:

```python
IN = ("frame",)           # ← index level names
CN = ("behaviour", "outcomes")  # ← column level names
```

Need semantic access to one name? Module constant. `FRAME = "frame"`. One import.

```python
from behavysis.df_classes.constants import FRAME
df.index.get_level_values(FRAME)  # ← clean, direct, no .value ceremony
```

**Karpathy #2**: "No abstractions for single-use code." The enum adds zero behavioural value — just extra lines. A Python `tuple` is what pandas MultiIndex names API actually wants.

**Remove**: `_enum_values()` entirely. Dead function. Its only purpose is confirming you picked wrong container.

---

### 2. Class explosion: 7 files for what is 2-3 lines of config each

| File                                 | Effective code                               |
| ------------------------------------ | -------------------------------------------- |
| `analysis_df.py` (23 lines)          | `IN=FramesIN, CN=AnalysisCN, NULLABLE=False` |
| `features_df.py` (20 lines)          | `IN=FramesIN, CN=FeaturesCN, NULLABLE=False` |
| `analysis_combined_df.py` (22 lines) | `IN=FramesIN, CN=AnalysisCombinedCN`         |
| `analysis_collated_df.py` (50 lines) | Two stubs, zero methods                      |

These are **data definitions**, not classes. Inheritance gives them `read`/`write`, but any function pair `read_keypoints(fp) -> pd.DataFrame`, `write_keypoints(df, fp)` would do same without class.

**Karpathy #2**: "If you write 200 lines and it could be 50, rewrite it." The df_classes dir is ~650 lines. 80% is enum definitions + class stubs.

---

### 3. `clean_and_validate` called redundantly

```
behav_df.py:112  init_df() → returns through clean_and_validate
behav_df.py:115  then caller clean_and_validates again
behav_df.py:228  bouts2frames → clean_and_validate at end
behav_df.py:259  fps_scale → clean_and_validate twice (line 254 + 260)
```

Sorting happens on every call. Sorting is O(n log n). Multiply by pipeline length, get real waste.

---

### 4. `NULLABLE` flag is barely used, inconsistently enforced

- `KeypointsDf._validate` calls `super()._validate(df)` — fine.
- `BehavDf._validate` calls `super()._validate(df)` — fine.
- But `_validate` is overridden in 3 subclasses. If someone forgets `super()` call, NaN check silently skipped.
- Many classes don't override `_validate` at all — `NULLABLE=False` is the only config differentiator. That's not enough justification for subclass.

---

### 5. Type annotation wrong: `EnumType | None = None`

`EnumType` is the metaclass. What you mean is `type[Enum] | None` or just `Any`. But subclass `IN` bound to actual enum _class_, not `EnumType`. Annotation lies.

---

## Proposed Surgery

### Phase 1: Kill Enum ceremonial overhead

```python
# Instead of:
class FramesIN(Enum):
    FRAME = "frame"

class KeypointsDf(DFMixin):
    IN = FramesIN

# Do:
FRAME = "frame"
KEYPOINTS_IN = ("frame",)
KEYPOINTS_CN = ("scorer", "individuals", "bodyparts", "coords")

class KeypointsDf(DFMixin):
    IN = KEYPOINTS_IN
    CN = KEYPOINTS_CN
```

Then access changes from `cls.IN.FRAME.value` → `FRAME` (direct string) or `cls.IN[0]` (index level by position). The 33 `cls.IN.X.value` / `cls.CN.X.value` calls across codebase all collapse.

### Phase 2: Merge thin config-only classes

```python
# one file: df_schemas.py
SCHEMAS = {
    "keypoints": {"in": ("frame",), "cn": ("scorer", "individuals", "bodyparts", "coords"), "nullable": False},
    "features":  {"in": ("frame",), "cn": ("features",), "nullable": False},
    "analysis":  {"in": ("frame",), "cn": ("individuals", "measures"), "nullable": False},
    # ...
}
```

Then `DFMixin` (or just a registry function) looks up schema by key. No 7 subclasses needed.

### Phase 3: `DFMixin` → function pair + validator

```python
def read_df(fp: Path, schema: tuple[tuple, tuple], fmt="parquet") -> pd.DataFrame: ...
def write_df(df: pd.DataFrame, fp: Path, schema, fmt="parquet") -> None: ...
def validate_schema(df: pd.DataFrame, idx_names, col_names, nullable=True) -> pd.DataFrame: ...
```

Inheritance replaced with composition. No class hierarchy. No `cls.` ceremony. No `super()` chain.

---

## Bottom line

You built a type-safe enum fortress around pandas strings. Pandas doesn't care about your enums — it wants tuples of strings. All the `_enum_values()` noise, `.value` ceremony, and 7 stub subclasses exist solely to serve an abstraction that adds nothing.

**Keep**: `CoordsCols`, `IndivCols`, `OutcomesPredictedCols`, `BoutCols` — these enums serve real value (discriminated unions for x/y/likelihood, predicted/scored column sets). They control logic paths.

**Kill**: `IN`/`CN` enums, all stub-only classes, `_enum_values()`, `EnumType` import.

Want me to draft the surgical plan for any of these phases?

---
