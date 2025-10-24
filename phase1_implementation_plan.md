# Phase 1 Implementation Plan: Critical Fixes & Foundation

## Overview
This plan addresses the most critical issues identified in the Behavysis codebase with immediate, high-impact fixes that can be implemented within 2-4 weeks.

## Priority 1: Error Handling & Resilience

### 1.1 Comprehensive Error Hierarchy
```python
# File: behavysis/core/errors.py
class BehavysisError(Exception):
    """Base exception for all Behavysis errors"""
    pass

class ConfigurationError(BehavysisError):
    """Configuration-related errors"""
    pass

class ProcessingError(BehavysisError):
    """Pipeline processing errors"""
    pass

class DLCIntegrationError(ProcessingError):
    """DeepLabCut integration errors"""
    pass

class DataValidationError(BehavysisError):
    """Data validation and integrity errors"""
    pass
```

### 1.2 Error Handling in Critical Methods

**File: [`behavysis/pipeline/project.py`](behavysis/pipeline/project.py:270-307)**
```python
def run_dlc(self, gputouse: int | None = None, overwrite: bool = False) -> None:
    """
    Batch processing with comprehensive error handling
    """
    try:
        # If gputouse is not specified, using all GPUs
        gputouse_ls = get_gpu_ids() if gputouse is None else [gputouse]
        if not gputouse_ls:
            raise DLCIntegrationError("No available GPUs found for DLC processing")
        
        nprocs = len(gputouse_ls)
        exp_ls = self.experiments
        
        if not overwrite:
            exp_ls = [exp for exp in exp_ls if not os.path.isfile(exp.get_fp(Folders.KEYPOINTS.value))]
        
        if not exp_ls:
            self.logger.info("No experiments require DLC processing")
            return
        
        exp_batches_ls = np.array_split(np.array(exp_ls), nprocs)
        
        with cluster_process(LocalCluster(n_workers=nprocs, threads_per_worker=1)):
            f_d_ls = [
                dask.delayed(RunDLC.ma_dlc_run_batch)(
                    vid_fp_ls=[exp.get_fp(Folders.FORMATTED_VID.value) for exp in exp_batch],
                    keypoints_dir=os.path.join(self.root_dir, Folders.KEYPOINTS.value),
                    configs_dir=os.path.join(self.root_dir, Folders.CONFIGS.value),
                    gputouse=gputouse,
                    overwrite=overwrite,
                )
                for gputouse, exp_batch in zip(gputouse_ls, exp_batches_ls)
            ]
            
            try:
                results = list(dask.compute(*f_d_ls))
                # Validate results and log diagnostics
                self._process_dlc_results(results)
                
            except Exception as e:
                raise DLCIntegrationError(f"DLC batch processing failed: {e}") from e
                
    except DLCIntegrationError:
        raise
    except Exception as e:
        raise ProcessingError(f"Unexpected error in DLC processing: {e}") from e
```

### 1.3 Input Validation Decorators
```python
# File: behavysis/utils/validation.py
from typing import Any, Callable, TypeVar
from functools import wraps

T = TypeVar('T')

def validate_configs(func: Callable[..., T]) -> Callable[..., T]:
    """Validate configuration files exist and are readable"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        configs_fp = kwargs.get('configs_fp') or args[2]  # Assuming configs_fp is 3rd arg
        if not os.path.isfile(configs_fp):
            raise ConfigurationError(f"Config file not found: {configs_fp}")
        if not os.access(configs_fp, os.R_OK):
            raise ConfigurationError(f"Config file not readable: {configs_fp}")
        return func(*args, **kwargs)
    return wrapper

def validate_dataframe_shape(func: Callable[..., T]) -> Callable[..., T]:
    """Validate DataFrame has expected shape and structure"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        df = kwargs.get('df') or args[0]
        if df.empty:
            raise DataValidationError("DataFrame is empty")
        if len(df) < 10:  # Minimum reasonable data points
            self.logger.warning("DataFrame has very few rows")
        return func(*args, **kwargs)
    return wrapper
```

## Priority 2: Testing Infrastructure

### 2.1 Test Framework Setup
```python
# File: test/conftest.py
import pytest
import tempfile
import shutil
from pathlib import Path

@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_configs():
    """Provide sample configuration for testing"""
    return {
        "user": {
            "format_vid": {"width_px": 960, "height_px": 540, "fps": 15},
            "run_dlc": {"model_path": "test_model"},
        },
        "auto": {},
        "ref": {}
    }

@pytest.fixture
def mock_experiment():
    """Create a mock experiment for testing"""
    class MockExperiment:
        def __init__(self):
            self.name = "test_experiment"
            self.root_dir = "/fake/path"
    
    return MockExperiment()
```

### 2.2 Critical Unit Tests
```python
# File: test/test_error_handling.py
import pytest
from behavysis.core.errors import (
    BehavysisError, ConfigurationError, ProcessingError, DLCIntegrationError
)
from behavysis.pipeline.project import Project

class TestErrorHandling:
    def test_project_initialization_with_invalid_dir(self):
        """Test Project initialization with non-existent directory"""
        with pytest.raises(ConfigurationError):
            Project("/non/existent/directory")
    
    def test_dlc_processing_no_gpus(self, monkeypatch):
        """Test DLC processing when no GPUs are available"""
        def mock_get_gpu_ids():
            return []
        
        monkeypatch.setattr("behavysis.utils.multiproc_utils.get_gpu_ids", mock_get_gpu_ids)
        
        project = Project(temp_project_dir)
        with pytest.raises(DLCIntegrationError):
            project.run_dlc()
    
    def test_config_validation_decorator(self):
        """Test configuration validation decorator"""
        @validate_configs
        def test_func(configs_fp: str):
            return "success"
        
        with pytest.raises(ConfigurationError):
            test_func("/invalid/path/config.json")
```

### 2.3 Integration Tests
```python
# File: test/test_pipeline_integration.py
class TestPipelineIntegration:
    def test_experiment_import_flow(self, temp_project_dir):
        """Test complete experiment import flow"""
        project = Project(temp_project_dir)
        
        # Create mock experiment files
        self._create_mock_experiment_files(temp_project_dir)
        
        project.import_experiments()
        assert len(project.experiments) == 1
        assert project.experiments[0].name == "test_experiment"
    
    def test_config_update_flow(self, temp_project_dir, sample_configs):
        """Test configuration update flow"""
        project = Project(temp_project_dir)
        project.update_configs("default_configs.json", "set")
        
        # Verify configs were updated
        configs = project.experiments[0].load_configs()
        assert configs.user.format_vid.width_px == 960
```

## Priority 3: Documentation & Type Hints

### 3.1 Complete Docstrings
**File: [`behavysis/pipeline/experiment.py`](behavysis/pipeline/experiment.py:127-176)**
```python
def _proc_scaff(
    self,
    funcs: tuple[Callable, ...],
    *args: Any,
    **kwargs: Any,
) -> dict[str, str]:
    """
    Process experiment using the provided functions with comprehensive error handling
    and diagnostics collection.

    This method serves as the central processing scaffold for all experiment operations,
    ensuring consistent error handling, logging, and diagnostics collection.

    Parameters
    ----------
    funcs : tuple[Callable, ...]
        Tuple of functions to execute in sequence. Each function should accept
        the same arguments and return a string outcome description.
    *args : Any
        Positional arguments to pass to each function.
    **kwargs : Any
        Keyword arguments to pass to each function.

    Returns
    -------
    dict[str, str]
        Diagnostics dictionary containing:
        - 'experiment': Experiment name
        - For each function: Function name as key, outcome description as value

    Raises
    ------
    ProcessingError
        If any function in the processing chain fails

    Examples
    --------
    >>> experiment._proc_scaff(
    ...     (FormatVid.format_vid, FormatVid.get_vid_metadata),
    ...     overwrite=True
    ... )
    {'experiment': 'test_exp', 'format_vid': 'Success', 'get_vid_metadata': 'Success'}
    """
    f_names_ls_msg = "".join([f"\n    - {f.__name__}" for f in funcs])
    self.logger.info(f"Processing experiment, {self.name}, with:{f_names_ls_msg}")
    
    dd = {"experiment": self.name}
    
    for f in funcs:
        f_name = f.__name__
        f_logger, f_io_obj = init_logger_io_obj(f_name)
        
        try:
            f(*args, **kwargs)
            dd[f_name] = get_io_obj_content(f_io_obj)
            
        except Exception as e:
            error_msg = f"Function {f_name} failed: {str(e)}"
            f_logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            dd[f_name] = f"ERROR: {error_msg}"
            
            # Re-raise as ProcessingError for upstream handling
            raise ProcessingError(error_msg) from e
            
        finally:
            f_io_obj.truncate(0)
    
    self.logger.info(f"Finished processing experiment, {self.name}, with:{f_names_ls_msg}")
    return dd
```

### 3.2 Enhanced Type Hints
```python
# File: behavysis/core/types.py
from typing import TypeVar, Union, Optional, Dict, List, Tuple
from pathlib import Path

# Type aliases for better readability
FilePath = Union[str, Path]
ExperimentName = str
BehaviorLabel = str
Probability = float
FrameNumber = int

# Generic types for DataFrame operations
DataFrameType = TypeVar('DataFrameType', bound='pd.DataFrame')
SeriesType = TypeVar('SeriesType', bound='pd.Series')

# Configuration types
ConfigDict = Dict[str, Union[str, int, float, bool, List, Dict]]
ValidationResult = Tuple[bool, Optional[str]]  # (is_valid, error_message)
```

## Priority 4: Performance Monitoring

### 4.1 Basic Performance Tracking
```python
# File: behavysis/utils/performance.py
import time
from contextlib import contextmanager
from typing import Optional

@contextmanager
def track_performance(operation_name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager to track operation performance and log timing information.
    
    Parameters
    ----------
    operation_name : str
        Name of the operation being tracked
    logger : Optional[logging.Logger]
        Logger instance for output, uses root logger if None
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        log_msg = f"Operation '{operation_name}' completed in {duration:.2f}s"
        
        if logger:
            logger.info(log_msg)
        else:
            logging.getLogger().info(log_msg)

# Usage example in critical methods
def format_vid(self, overwrite: bool) -> dict:
    with track_performance("video_formatting", self.logger):
        return self._proc_scaff(
            (FormatVid.format_vid,),
            raw_vid_fp=self.get_fp(Folders.RAW_VID),
            formatted_vid_fp=self.get_fp(Folders.FORMATTED_VID),
            configs_fp=self.get_fp(Folders.CONFIGS),
            overwrite=overwrite,
        )
```

## Implementation Timeline

### Week 1: Foundation
- [ ] Create error hierarchy and base exceptions
- [ ] Implement validation decorators
- [ ] Set up test framework with pytest
- [ ] Create basic unit tests for error handling

### Week 2: Core Integration
- [ ] Integrate error handling in Project class methods
- [ ] Add comprehensive input validation
- [ ] Implement performance tracking
- [ ] Create integration tests for pipeline flows

### Week 3: Documentation & Polish
- [ ] Complete docstrings for all public methods
- [ ] Enhance type hints throughout codebase
- [ ] Create usage examples and documentation
- [ ] Performance benchmarking and optimization

### Week 4: Validation & Deployment
- [ ] Run comprehensive test suite
- [ ] Performance testing with real datasets
- [ ] Update documentation with new features
- [ ] Create migration guide for existing users

## Success Metrics

### Code Quality
- **Test Coverage**: >80% for core modules
- **Type Hint Coverage**: 100% for public interfaces
- **Documentation Coverage**: 100% for public methods

### Reliability
- **Error Handling**: All critical paths have proper exception handling
- **Input Validation**: All public methods validate inputs
- **Graceful Degradation**: System continues operating when non-critical components fail

### Performance
- **Memory Usage**: No memory leaks in long-running processes
- **Processing Time**: <10% overhead from new error handling
- **Resource Utilization**: Proper cleanup of temporary resources

## Risk Mitigation

### Backward Compatibility
- All changes maintain existing API interfaces
- Deprecated methods marked with warnings
- Comprehensive migration documentation

### Testing Strategy
- Unit tests for all new functionality
- Integration tests for critical workflows
- Performance regression tests
- Compatibility tests with existing projects

This Phase 1 implementation establishes a solid foundation for the long-term health and scalability of the Behavysis codebase while addressing the most critical issues immediately.