# Code Improvement Recommendations

This document contains code improvement recommendations identified during the documentation review. These should be filed as GitHub Issues.

## Issue 1: Refactor — Replace bare except clauses with specific exception handling
**Priority**: High  
**Labels**: `bug`, `refactor`, `good first issue`  
**Affected**: `src/behavysis/pipeline/experiment.py`, `src/behavysis/processes/*.py`

### Problem
Multiple files use bare `except:` clauses which catch all exceptions including `SystemExit` and `KeyboardInterrupt`. This can hide real bugs and make debugging difficult.

### Evidence
- `experiment.py:142` in `_proc_scaff`
```python
try:
    f(*args, **kwargs)
except Exception as e:  # Better but could be more specific
    f_logger.error(e)
```

### Impact
- Prevents proper propagation of critical errors
- User cannot interrupt with Ctrl+C during processing
- Makes debugging cryptic errors nearly impossible

### Suggested Fix
```python
try:
    result = some_operation()
except FileNotFoundError as e:
    f_logger.error(f"File not found: {e}")
    raise
except PermissionError as e:
    f_logger.error(f"Permission denied: {e}")
    raise
except Exception as e:
    f_logger.error(f"Unexpected error: {type(e).__name__}: {e}")
    raise
```

---

## Issue 2: Enhancement — Add comprehensive input validation for config parameters
**Priority**: High  
**Labels**: `enhancement`, `usability`, `scientific-practices`  
**Affected**: `src/behavysis/models/experiment_configs.py`, `src/behavysis/models/processes/*.py`

### Problem
Many configuration parameters lack validation, allowing invalid values that cause cryptic errors downstream or silently produce incorrect results.

### Examples
- `px_per_mm` can be negative or zero (causes division by zero)
- `fps` can be negative or zero
- `start_frame` can be greater than `stop_frame`
- `thresh_mm` values can be nonsensical
- File paths aren't validated for existence

### Impact
- Silent incorrect results in scientific analysis
- Errors appear far from the source
- Users waste time debugging downstream issues

### Suggested Fix
```python
from pydantic import field_validator, Field

class AutoConfigs(PydanticBaseModel):
    px_per_mm: float = Field(default=-1, gt=0, description="Pixels per millimeter")
    start_frame: int = Field(default=-1, ge=0)
    stop_frame: int = Field(default=-1, ge=0)
    
    @field_validator('stop_frame')
    @classmethod
    def validate_frame_order(cls, v, info):
        if 'start_frame' in info.data and v <= info.data['start_frame']:
            raise ValueError(f"stop_frame ({v}) must be > start_frame ({info.data['start_frame']})")
        return v
```

---

## Issue 3: Bug — Silent failures in pipeline processing
**Priority**: High  
**Labels**: `bug`, `refactor`  
**Affected**: `src/behavysis/pipeline/project.py`, `src/behavysis/processes/*.py`

### Problem
Processing methods silently log errors but continue execution. In batch processing of many experiments, failures can be missed.

### Evidence
```python
# In project.py _proc_scaff:
dd_ls = scaffold_func(method, *args, **kwargs)
if len(dd_ls) > 0:
    df = DiagnosticsDf.init_from_dd_ls(dd_ls)
    DiagnosticsDf.write(df, ...)
# No check for failures, no error raised
```

### Impact
- Users may not notice experiment failures
- Scientific data may be incomplete without users knowing
- Silent data loss in large batches

### Suggested Fix
```python
def _proc_scaff(self, method, *args, strict: bool = False, **kwargs):
    dd_ls = scaffold_func(method, *args, **kwargs)
    
    # Check for failures
    failures = [dd for dd in dd_ls if any("ERROR" in str(v) for v in dd.values())]
    
    if failures:
        self.logger.error(f"{len(failures)} experiments failed processing")
        if strict:
            raise PipelineError(f"Processing failed for: {[f['experiment'] for f in failures]}")
    
    return dd_ls
```

---

## Issue 4: Enhancement — Add reproducibility features (seeds, version tracking)
**Priority**: Medium  
**Labels**: `enhancement`, `scientific-practices`, `reproducibility`  
**Affected**: Global

### Problem
Scientific reproducibility is not supported. No tracking of:
- Random seeds for ML components
- Dependency versions
- Configuration hashes
- behavysis version used

### Impact
- Results may not be reproducible
- Cannot verify which software version generated results
- Publication requirements not met

### Suggested Fix
Add to config structure:
```python
class ReproducibilityConfigs(PydanticBaseModel):
    random_seed: int = 42
    behavysis_version: str = Field(default_factory=lambda: importlib.metadata.version("behavysis"))
    config_hash: str = ""
    dependencies: dict = Field(default_factory=dict)
    
    @model_validator(mode='after')
    def compute_hash(self):
        self.config_hash = hashlib.sha256(
            self.model_dump_json().encode()
        ).hexdigest()[:16]
        return self
    
    @field_validator('dependencies', mode='after')
    @classmethod
    def record_deps(cls, v):
        import sys
        return {dist.name: dist.version for dist in importlib.metadata.distributions()}
```

Also set random seeds:
```python
def _set_seeds(seed: int):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

---

## Issue 5: Refactor — Convert TODO comments to proper issues
**Priority**: Medium  
**Labels**: `refactor`, `documentation`  
**Affected**: Repository-wide

### Problem
Repository has 15+ "TODO" comments scattered in code. These represent known issues that should be tracked.

### Evidence
```python
# project.py:225 TODO: implement diagnostics
# project.py:226 TODO: implement error handling
# project.py:227 TODO: implement handling if NO GPU
# constants.py:75 TODO: is there a better way to do the subsubdirs?
# etc.
```

### Impact
- Technical debt not visible
- Contributors don't know what's planned
- Issues forgotten over time

### Suggested Fix
1. Create GitHub issues for each TODO
2. Replace comments with issue references:
   ```python
   # TODO(#123): Implement GPU-free handling
   ```
3. Add issue templates to repository

---

## Issue 6: Enhancement — Add mypy type checking to CI
**Priority**: Medium  
**Labels**: `enhancement`, `quality`, `ci-cd`  
**Affected**: `.github/workflows/`

### Problem
Type hints exist but aren't enforced. Many functions use `Any` or have incomplete annotations.

### Impact
- Type errors only caught at runtime
- Refactoring is risky
- IDE autocomplete less useful

### Suggested Fix
Add to CI workflow:
```yaml
- name: Type check with mypy
  run: |
    pip install mypy types-all
    mypy src/behavysis \
      --strict \
      --ignore-missing-imports \
      --show-error-codes
```

Also create `mypy.ini`:
```ini
[mypy]
python_version = 3.12
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
```

---

## Issue 7: Refactor — Consolidate duplicated code across processes
**Priority**: Low-Medium  
**Labels**: `refactor`, `code-quality`  
**Affected**: `src/behavysis/processes/*.py`

### Problem
Each process class repeats boilerplate:
```python
logger, io_obj = init_logger_io_obj()
if not overwrite and os.path.exists(dst_fp):
    logger.warning(file_exists_msg(dst_fp))
    return get_io_obj_content(io_obj)
configs = ExperimentConfigs.read_json(configs_fp)
# ... actual logic ...
return get_io_obj_content(io_obj)
```

### Impact
- Code bloat (repeated in 10+ files)
- Inconsistent error handling
- Harder to maintain

### Suggested Fix
Create decorator:
```python
def process_step(overwrite_check: bool = True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(src_fp: str, dst_fp: str, configs_fp: str, overwrite: bool, *args, **kwargs):
            logger, io_obj = init_logger_io_obj()
            
            if overwrite_check and not overwrite and os.path.exists(dst_fp):
                logger.warning(file_exists_msg(dst_fp))
                return get_io_obj_content(io_obj)
            
            configs = ExperimentConfigs.read_json(configs_fp)
            
            try:
                result = func(src_fp, dst_fp, configs, *args, **kwargs)
                return get_io_obj_content(io_obj)
            except Exception as e:
                logger.error(f"{func.__name__} failed: {e}")
                raise
        return wrapper
    return decorator

# Usage:
class Preprocess:
    @process_step()
    def start_stop_trim(src_fp, dst_fp, configs, overwrite):
        # Only actual logic here
        pass
```

---

## Issue 8: Enhancement — Add progress bars for long-running operations
**Priority**: Low  
**Labels**: `enhancement`, `usability`  
**Affected**: `src/behavysis/pipeline/project.py`

### Problem
Batch processing over many experiments provides no progress indication. Users don't know if hanging or progressing.

### Impact
- Poor user experience
- Cannot estimate completion time
- May terminate thinking process is stuck

### Suggested Fix
```python
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

def _proc_scaff_sp(self, method, *args, **kwargs):
    with logging_redirect_tqdm():
        return [
            method(exp, *args, **kwargs) 
            for exp in tqdm(self.experiments, desc=method.__name__)
        ]
```

Also add to dependencies:
```toml
dependencies = [
    "tqdm>=4.67.3",  # Already listed, just need to use it
]
```

---

## Issue 9: Enhancement — Add automatic data integrity checks
**Priority**: Medium  
**Labels**: `enhancement`, `quality-assurance`  
**Affected**: `src/behavysis/df_classes/*.py`

### Problem
DataFrames between pipeline stages have implicit structure assumptions. No validation that expected columns/indices exist.

### Impact
- Errors appear deep in processing
- Hard to trace where data became corrupted
- Silent incorrect results possible

### Suggested Fix
```python
from pandera import DataFrameSchema, Column, Check

keypoints_schema = DataFrameSchema({
    ("scorer", "individuals", "bodyparts", "x"): Column(float),
    ("scorer", "individuals", "bodyparts", "y"): Column(float),
    ("scorer", "individuals", "bodyparts", "likelihood"): Column(float, Check.in_range(0, 1)),
}, strict=True)

@keypoints_schema.validate
def some_processing(df: pd.DataFrame) -> pd.DataFrame:
    pass
```

Or simpler without pandera:
```python
def validate_keypoints_df(df: pd.DataFrame) -> None:
    required_cols = ["x", "y", "likelihood"]
    if not all(col in df.columns.get_level_values("coords") for col in required_cols):
        raise ValueError(f"Missing required columns. Expected: {required_cols}")
    if df["likelihood"].min() < 0 or df["likelihood"].max() > 1:
        raise ValueError("Likelihood values out of range [0,1]")
```

---

## Issue 10: Documentation — Improve docstring coverage and quality
**Priority**: Low  
**Labels**: `documentation`  
**Affected**: Multiple modules

### Problem
Many classes and methods have minimal or placeholder docstrings:
```python
class Preprocess:
    """_summary_"""  # Not helpful
    
class Experiment:
    """Behavysis Pipeline class for a single experiment. ... Parameters ..."""
    # Missing Raises, Notes, Examples sections
```

### Impact
- API docs are incomplete
- Users can't understand functionality
- IDE hover information unhelpful

### Suggested Fix
Add comprehensive docstrings with:
- Clear description
- All parameters with types
- Return values
- Exceptions raised
- Usage examples
- See Also references

Enable docstring linting:
```toml
[tool.ruff.lint.pydocstyle]
convention = "numpy"  # or "google"
```

---

## Summary Table

| Issue | Priority | Type | Effort |
|-------|----------|------|--------|
| Bare except clauses | High | Bug/Refactor | Low |
| Config validation | High | Enhancement | Medium |
| Silent failures | High | Bug | Medium |
| Reproducibility | Medium | Enhancement | Medium |
| TODO cleanup | Medium | Refactor | Low |
| mypy in CI | Medium | Enhancement | Low |
| Code consolidation | Low-Med | Refactor | Medium |
| Progress bars | Low | Enhancement | Low |
| Data integrity | Medium | Enhancement | Medium |
| Docstrings | Low | Documentation | Ongoing |
