# Behavysis pipeline paradigm assessment

**Session ID:** ses_0e12e92bcffeBN3MMWAow2eRZs
**Created:** 7/2/2026, 3:54:29 AM
**Updated:** 7/2/2026, 3:57:35 AM

---

## User

You are a principle data scientist and data engineer.
Be critical, verify what you do, be elegant in your solutions, be honest and harsh but fair. use your critical thinking and validating your thinking with checks. Use karpathy guidelines.

Context: There's a lot of sprawl for @src/behavysis/schemas/ @src/behavysis/models/ and @src/behavysis/utils/ . Are they all needed? Am I doing things in a round-about way? Could it be better?
Is the @src/behavysis/pipeline/experiment.py and @src/behavysis/pipeline/project.py paradigms good to boot as well?
What about how logging does not raise an error in trace?

Your task: Critically assess the viability of this current paradigm. Research other paradigms that are a better fit. Be effective, be elegant, be creative. Present your assessment.

---

# Critical Architecture Assessment: `behavysis`

## Executive verdict

The **paradigm is sound** — a staged, filesystem-backed pipeline with plugin analysis functions layered on Polars long-form DataFrames. This is the right design for animal behaviour video analysis. The problems are **discipline and misplaced code**, not failed paradigms.

---

## 🔴 CRITICAL: `@trace` silently swallows all exceptions

```python
# logger_utils.py:98-101
except Exception:
    elapsed = time.perf_counter() - t0
    logger.exception("✗ FAILED after {:.2f}s", elapsed)
    # Don't raise so pipeline doesn't stop for other experiments
```

**This is the single biggest bug in the codebase.** Every `@trace`-decorated method (all 12 pipeline stages in `Experiment`) returns `None` on failure. Downstream code then crashes with cryptic `AttributeError: 'NoneType' object has no attribute 'select'` — the real root cause is buried in a log file.

**The fix**: `@trace` must **re-raise** always. Batch resilience belongs in `Project._run_sequential`:

```python
def _run_sequential(self, method, **kwargs):
    errors = []
    for exp in self.experiments:
        try:
            method(exp, **kwargs)
        except Exception:
            logger.exception(f"Experiment '{exp.name}' failed")
            errors.append(exp.name)
    if errors:
        logger.error(f"{len(errors)} experiments failed: {errors}")
```

The decorator is a pure observability tool, not a control flow mechanism.

---

## 🟠 MAJOR: `schemas/` has an identity crisis

```
schemas/
├-- schemas.py          ← Schema dicts + read_df/write_df  (182 lines) ✅
├-- analysis_agg.py     ← agg_quantitative, make_binned, summary_binned (348 lines) ❌ NOT a schema
├-- behaviour.py        ← vect2bouts, bouts2frames, predicted2scored (362 lines) ❌ NOT a schema
└-- keypoints.py        ← check_bpts_exist, get_indivs_bpts (49 lines)    ❌ NOT a schema
```

710 of 941 lines in the `schemas/` package are **domain transformation functions**, not schemas. `from behavysis.schemas import vect2bouts` is semantically wrong — the caller is importing business logic from what it believes is a schema definitions module.

**The fix**:

- `schemas.py` stays as `schemas/` (schema dicts + I/O validation at the boundary)
- `behaviour.py` → `funcs/behaviour_transforms.py` (bout detection, frame↔bout conversion, BORIS import)
- `analysis_agg.py` → `funcs/analysis_transforms.py` or merge into `funcs/analyse/`
- `keypoints.py` → `funcs/keypoint_utils.py`

---

## 🟠 MAJOR: `configure_logger()` called at import time

```python
# __init__.py:28
configure_logger()  # Called unconditionally on `import behavysis`
```

This is a side-effect anti-pattern. Anyone who imports `behavysis` gets their loguru handlers wiped (`logger.remove()` on line 31). If a downstream library also uses loguru (e.g. DLC, SimBA), this silently destroys their logging config.

**The fix**: Remove from `__init__.py`. Make it an explicit setup call the user runs in their pipeline script. The `run_pipeline_script.py` template already has a natural place for this.

---

## 🟡 MEDIUM: `models/__init__.py` has a dormant bug

```python
# models/__init__.py:19
"AnalysisConfig",  # This class does not exist. Actual class is AnalyseConfig.
```

Currently harmless (no wildcard imports from this module), but it's a time bomb. Any `from behavysis.models import *` will raise `AttributeError`.

---

## 🟡 MEDIUM: Boilerplate proliferation

**16 `require_*()` methods** across `ExperimentConfig` (7), `ExperimentMetadata` (9), and `SubfuncModel` (a generic `require()` that's inconsistently used):

```python
def require_name(self) -> str:
    if self.name is None: raise MetadataNotReadyError("name", "...")
    return self.name
# ... repeated 15 more times with different field names
```

**The fix**: Replace with a descriptor or a single generic method:

```python
from pydantic import model_validator

class ExperimentMetadata(BaseModel):
    @classmethod
    def require[T](cls, instance: "ExperimentMetadata", field: str, stage: str) -> T:
        val = getattr(instance, field)
        if val is None:
            raise MetadataNotReadyError(field, stage)
        return val
```

Or just make the pipeline stages set these eagerly instead of lazy-checking — if `format_video()` always populates `formatted_video.fps`, then `require_fps()` is just `return self.formatted_video.fps` with confidence.

---

## 🟡 MEDIUM: Project `collate_*` methods inline raw schema dicts

```python
# project.py:259-266, 303-309
write_df(combined, out_fp, {
    "bin_sec": pl.Float64,
    "experiment": pl.Utf8,
    ...
})
```

These are the **defined schemas** `COLLATED_BINNED_SCHEMA` and `COLLATED_SUMMARY_SCHEMA` from `schemas.py`. The inlined dicts are a DRY violation that will cause schema drift bugs. Use the defined constants.

---

## 🟡 MEDIUM: `classify_behaviour` parallelization hack

```python
# project.py:177-178
nprocs = self.nprocs
self.nprocs = 1   # "Temporarily use single processing due to IO issues"
```

Mutating instance state in a method as a workaround is fragile. If an exception occurs before the restore (line 183), `self.nprocs` stays permanently changed. This indicates the parallel decomposition needs reconsideration for I/O-bound stages.

---

## 🔵 LOWER: Bare `print()` calls in library code

```python
# dask_utils.py:15
print(client.dashboard_link)  # Debug noise in library context

# template_utils.py:20
print("Please enter 'y' or 'n'.")  # Bypasses logging framework
```

Both should use `logger.info()` or be removed. The template_utils one is defensible (CLI prompt), but the dask_utils one is debug spam.

---

## 🔵 LOWER: 88 string constants in `constants/data_names.py`

`X = "x"` adds indirection without type safety. The schemas are where these earn their keep (cross-referenced in validation). But many are single-use. Consider only exporting the ones used in schemas and letting Polars accept inline strings elsewhere.

---

## ✅ What's good

| Aspect                                                                                  | Assessment                                                                                                                                                                                                             |
| --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Polars long-form schema design**                                                      | Excellent. The migration from pandas MultiIndex was the right call. Schema validation at I/O boundaries is disciplined.                                                                                                |
| **Plugin protocol system** (`PreprocessFunc`, `AnalyseFunc`, `CalculateParametersFunc`) | Elegant. Duck-typed callable protocols are the right abstraction. The `AnalysisResult` TODO in `_helper.py:36` suggests you're already aware the `AnalyseFunc` signature could return data instead of writing to disk. |
| **Filesystem-as-state pattern**                                                         | Pragmatic and correct. `Experiment.get_fp()` is simple, `overwrite` checks are consistent. This is how scientific pipelines should work.                                                                               |
| **Pydantic v2 config models**                                                           | Good validation, `read_yaml()` is clean. The `require_*()` pattern is the right idea even if the implementation is verbose.                                                                                            |
| **`Experiment` → `Project` separation**                                                 | Clear single-vs-batch boundary. `Project` delegates well.                                                                                                                                                              |
| **Dask for parallelism**                                                                | Appropriate for CPU-bound stages.                                                                                                                                                                                      |

---

## Alternative paradigms considered

1. **DAG orchestrator (Prefect/Dagster)**: Overkill for a 9-stage linear pipeline. Revisit if stages become non-linear (e.g. branching analyses, conditional execution).

2. **`sklearn.pipeline`-style functional composition**: You're already doing this — `PreprocessFunc` is exactly `f(df) -> df`. The `analyse` stage breaks the pattern (writes to disk), which you've already flagged with the `AnalysisResult` TODO.

3. **Event sourcing / artifact store**: You're already doing this via the filesystem. A more structured artifact store (e.g. MLflow, W&B) would give you provenance tracking for free, but adds operational complexity.

**My recommendation**: Stay with the current paradigm. It's well-fitted to the domain. Fix the discipline issues, and you'll have a solid foundation.

---

## Priority-ordered action items

| #   | Severity    | What                                                                        | Where                                                                     |
| --- | ----------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| 1   | 🔴 Critical | Make `@trace` re-raise; add try/except to `Project._run_sequential`         | `logger_utils.py`, `project.py`                                           |
| 2   | 🟠 Major    | Remove `configure_logger()` from `__init__.py`                              | `__init__.py`                                                             |
| 3   | 🟠 Major    | Split `schemas/` — move business logic to `funcs/`                          | `schemas/analysis_agg.py`, `schemas/behaviour.py`, `schemas/keypoints.py` |
| 4   | 🟡 Medium   | Fix `"AnalysisConfig"` in `models/__init__.py`                              | `models/__init__.py:19`                                                   |
| 5   | 🟡 Medium   | Use `COLLATED_BINNED_SCHEMA`/`COLLATED_SUMMARY_SCHEMA` in `collate_*`       | `project.py:256-267`, `project.py:299-310`                                |
| 6   | 🟡 Medium   | Consolidate 16 `require_*()` methods with a generic `require()` classmethod | `experiment_config.py`, `experiment_metadata.py`                          |
| 7   | 🔵 Lower    | Replace bare `print()` with `logger.info()`                                 | `dask_utils.py:15`                                                        |
| 8   | 🔵 Lower    | Fix `classify_behaviour` parallelization hack                               | `project.py:177-183`                                                      |

---
