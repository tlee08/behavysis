# Behavysis Implementation Guide

## Quick Start: Phase 1 Critical Fixes

### 1. Error Handling Foundation (Week 1)

**Create Error Hierarchy**
```python
# behavysis/core/errors.py
class BehavysisError(Exception): ...
class ConfigurationError(BehavysisError): ...
class ProcessingError(BehavysisError): ...
class DLCIntegrationError(ProcessingError): ...
class DataValidationError(BehavysisError): ...
```

**Fix Critical Methods**
- **File**: `behavysis/pipeline/project.py` (lines 270-307)
- **Problem**: Missing error handling in DLC processing
- **Solution**: Add try/except blocks and raise specific exceptions

**Add Input Validation**
```python
# behavysis/utils/validation.py
@validate_configs
def method(configs_fp: str): ...
```

### 2. Testing Setup (Week 2)

**Essential Tests**
```python
# test/test_error_handling.py
def test_project_invalid_dir(): ...
def test_dlc_no_gpus(): ...
def test_config_validation(): ...
```

**Test Configuration**
- Use pytest with fixtures in `test/conftest.py`
- Create `temp_project_dir` fixture for isolated testing
- Mock external dependencies (GPU detection, DLC)

### 3. Documentation & Types (Week 3)

**Complete Missing Docstrings**
- **File**: `behavysis/pipeline/experiment.py` (lines 127-176)
- **Add**: Parameters, Returns, Raises, Examples sections
- **Fix**: Replace `_summary_` placeholders

**Enhanced Type Hints**
```python
# behavysis/core/types.py
FilePath = Union[str, Path]
ExperimentName = str
ConfigDict = Dict[str, Union[str, int, float, bool, List, Dict]]
```

### 4. Performance Monitoring (Week 4)

**Add Basic Tracking**
```python
# behavysis/utils/performance.py
@contextmanager
def track_performance(operation_name: str): ...
```

**Integrate in Critical Methods**
- Add to `format_vid`, `run_dlc`, `classify_behavs` methods
- Log timing information for performance analysis

## Key Files to Modify

### High Priority
1. `behavysis/pipeline/project.py` - Add error handling to `run_dlc` method
2. `behavysis/pipeline/experiment.py` - Complete docstrings for `_proc_scaff`
3. `behavysis/behav_classifier/behav_classifier.py` - Fix memory issues in data concatenation

### Medium Priority
1. `behavysis/processes/classify_behavs.py` - Add input validation
2. `behavysis/models/experiment_configs.py` - Enhance configuration validation
3. `behavysis/utils/df_mixin.py` - Add data validation checks

## Critical Code Changes

### Error Handling in DLC Processing
```python
# In project.py run_dlc method
try:
    # Existing logic
    gputouse_ls = get_gpu_ids() if gputouse is None else [gputouse]
    if not gputouse_ls:
        raise DLCIntegrationError("No available GPUs")
    # ... rest of method
except DLCIntegrationError:
    raise
except Exception as e:
    raise ProcessingError(f"DLC processing failed: {e}") from e
```

### Input Validation Decorator
```python
def validate_configs(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        configs_fp = kwargs.get('configs_fp') or args[2]
        if not os.path.isfile(configs_fp):
            raise ConfigurationError(f"Config file not found: {configs_fp}")
        return func(*args, **kwargs)
    return wrapper
```

## Success Metrics

- **Test Coverage**: >80% for core modules
- **Error Handling**: All critical paths have proper exception handling
- **Documentation**: 100% of public methods documented
- **Performance**: <10% overhead from new error handling

## Quick Wins (First 2 Days)

1. **Create error hierarchy** in `behavysis/core/errors.py`
2. **Add basic tests** for Project initialization
3. **Fix critical docstrings** in experiment.py
4. **Add input validation** to public methods

This implementation plan addresses the most critical issues while maintaining backward compatibility and establishing a foundation for long-term codebase health.