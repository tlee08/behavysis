# Logging Architecture Assessment Report

## Branch: `refactor/logging-architecture`

---

## 1. Current Logging Architecture Analysis

### Overview
The current logging system is a custom wrapper around Python's standard `logging` module located in `src/behavysis/utils/logging_utils.py`.

### Key Components

#### logging_utils.py
```python
# Three handler types defined:
- add_console_handler()     → StreamHandler to sys.stderr
- add_log_file_handler()    → FileHandler to ~/.behavysis/debug.log
- add_io_obj_handler()      → StreamHandler to io.StringIO
```

#### Initialization Functions
```python
- init_logger_console()     → console only
- init_logger_file()        → console + file (used by Project, Experiment class logger)
- init_logger_io_obj()      → console + file + StringIO (used by process methods)
```

### Usage Patterns Found

1. **Class-level loggers** (Project, Experiment):
   ```python
   logger = init_logger_file()  # Single logger shared across all instances
   ```

2. **Process-level loggers** (all process methods):
   ```python
   logger, io_obj = init_logger_io_obj(func_name)
   # ... do work, log to logger ...
   return get_io_obj_content(io_obj)
   ```

3. **Diagnostics flow**:
   - Each process function returns a string from `get_io_obj_content(io_obj)`
   - Returned to `_proc_scaff()` in Experiment
   - Collected into a dictionary and saved to `0_diagnostics/{method_name}.csv`

### Three Target Outputs

| Target | Purpose | Current Implementation |
|--------|---------|----------------------|
| Console | User-facing progress messages | StreamHandler → sys.stderr |
| Debug logs (~/.behavysis/) | Persistent debug records | FileHandler → ~/.behavysis/debug.log |
| Diagnostics (./0_diagnostics/) | Per-process outcomes per experiment | io.StringIO → CSV files |

---

## 2. What's Broken or Not Working

### A. Debug Logs (~/.behavysis/)
**Status: ⚠️ SHARED SINGLE FILE ISSUE**

- All loggers write to **the same file**: `~/.behavysis/debug.log`
- Multiple `FileHandler` instances get added to different loggers
- **Duplicate log entries** occur because multiple handlers write to the same file
- No log rotation or file management
- The `add_log_file_handler()` checks for existing handlers by filename, but doesn't prevent ALL loggers writing to the same file

### B. Console Output
**Status: ✅ Works but has issues**

- Console logging does work
- However, it uses **sys.stderr instead of sys.stdout** which can confuse some consumers
- Duplicate console output when multiple loggers are initialized (each adds its own StreamHandler)
- `init_logger_io_obj()` adds handlers **every time it's called for the same logger name**

### C. Diagnostics Output
**Status: ✅ Works but over-complicated**

- Diagnostics CSV files are created successfully
- Each process method returns its StringIO content as a string
- The `_proc_scaff()` method aggregates these into dictionaries and saves to CSV
- **Problem**: The StringIO handler is never removed, causing memory leaks

### D. Logger Instance Issues
**Status: ⚠️ COMMON LOGGER NAMES**

```python
# In Experiment._proc_scaff():
f_logger, f_io_obj = init_logger_io_obj(f_name)  # uses function name

# But all experiments share the same logger if function names collide
# And Project/Experiment use class-level loggers
```

**Critical Issue**: The `init_logger_file()` is called once at class definition time, not instantiation. This means:
- All `Project` instances share one logger
- All `Experiment` instances share one logger
- Log messages from different instances are mixed together

---

## 3. Why It's Over-Complicated

### Example 1: Simple logging requires 3 functions
```python
# Current approach - need to import and call special init
from behavysis.utils.logging_utils import get_io_obj_content, init_logger_io_obj

def my_process():
    logger, io_obj = init_logger_io_obj()
    logger.info("Starting")
    # ... work ...
    logger.info("Done")
    return get_io_obj_content(io_obj)
```

**Should be:**
```python
import logging

def my_process():
    logger = logging.getLogger(__name__)  # Standard Python pattern
    logger.info("Starting")
    # ... work ...
    logger.info("Done")
    return {"status": "success", "messages": [...]}  # Structured return
```

### Example 2: io.StringIO as a handler is unnecessary
```python
# Current - captures logs via StringIO handler
def init_logger_io_obj(name=None):
    logger = init_logger(...)  # Creates logger with console, file, AND StringIO handlers
    io_obj = add_io_obj_handler(logger)  # Adds another handler
    return logger, io_obj

def get_io_obj_content(io_obj):
    # Manually reads from StringIO
    cursor = io_obj.tell()
    io_obj.seek(0)
    msg = io_obj.read()
    io_obj.seek(cursor)
    return msg
```

**Problems:**
1. The logger accumulates handlers every time it's called
2. StringIO grows indefinitely in memory
3. Complex manual cursor management
4. Takes string content and puts it in CSV (log format → cell content)

### Example 3: Function name from stack trace
```python
def init_logger_io_obj(name=None, ...):
    name = name or get_func_name_in_stack(2)  # Magic introspection
```

**Why this is bad:**
- Hard to trace where logger names come from
- Breaks if call stack changes
- Makes testing difficult
- Violates explicit is better than implicit

### Example 4: Multiple handler detection logic
```python
def add_console_handler(logger, level):
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            if handler.stream == sys.stderr:
                return  # Handler already exists
    # Create new handler...
```

**Problems:**
1. Each function has its own handler detection logic
2. Doesn't prevent multiple handlers of same type with different configs
3. O(n) scan on every call

### Example 5: Data flow for diagnostics is convoluted
```
Process method
  ↓ init_logger_io_obj() → creates logger with StringIO handler
  ↓ logs messages
  ↓ get_io_obj_content(io_obj) → extracts string
  ↓ returns string
  ↓ _proc_scaff() collects into dict
  ↓ dict saved to CSV
```

**Should be:**
```
Process method
  ↓ uses standard logger
  ↓ collects structured results in code
  ↓ returns structured dict
  ↓ saved to structured format (JSON/CSV)
```

---

## 4. Best-Practice Recommendations

### A. Use Python's Standard Logging Properly

```python
# In __init__.py or at module level
import logging

# Single configuration point
def setup_logging(
    console_level=logging.INFO,
    debug_file: Path | None = None,
    debug_level=logging.DEBUG
):
    """Configure root logging once per application."""
    root_logger = logging.getLogger("behavysis")
    root_logger.setLevel(logging.DEBUG)
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    root_logger.addHandler(console)
    
    # Optional file handler (one per project, not global)
    if debug_file:
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(debug_file)
        file_handler.setLevel(debug_level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        ))
        root_logger.addHandler(file_handler)

# In any module, just:
logger = logging.getLogger(__name__)
```

### B. Structured Logging Approach

```python
import json
from dataclasses import dataclass
from typing import Literal

@dataclass
class ProcessResult:
    experiment: str
    function: str
    status: Literal["success", "error", "skipped"]
    message: str
    details: dict | None = None
    
    def to_dict(self) -> dict:
        return {
            "experiment": self.experiment,
            "function": self.function,
            "status": self.status,
            "message": self.message,
            **(self.details or {})
        }

# In process methods:
def my_process(experiment: str, ...) -> ProcessResult:
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"Processing {experiment}")
        # ... work ...
        return ProcessResult(
            experiment=experiment,
            function="my_process",
            status="success",
            message="Completed successfully"
        )
    except Exception as e:
        logger.error(f"Failed: {e}")
        return ProcessResult(
            experiment=experiment,
            function="my_process",
            status="error",
            message=str(e),
            details={"traceback": traceback.format_exc()}
        )
```

### C. Clear Separation of Concerns

#### Console Output (User-facing)
- **Purpose**: Show user progress and results
- **Level**: INFO and above
- **Format**: Simple, readable
- **When**: All the time

```python
# Setup once
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
```

#### Debug Logs (~/.behavysis/)
- **Purpose**: Detailed debugging per project
- **Level**: DEBUG and above
- **Format**: Detailed with timestamps, file:line numbers
- **When**: Always written, developer reads when needed

```python
# Per-project debug log
debug_file = Path.home() / ".behavysis" / project_name / "debug.log"
debug_file.parent.mkdir(parents=True, exist_ok=True)
```

#### Diagnostics (./0_diagnostics/)
- **Purpose**: Machine-readable process outcomes
- **Format**: Structured JSON or CSV with schema
- **When**: Generated per batch operation

```python
def save_diagnostics(results: list[ProcessResult], output_dir: Path):
    """Save structured diagnostics to file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV for tabular data
    df = pd.DataFrame([r.to_dict() for r in results])
    df.to_csv(output_dir / "batch_results.csv", index=False)
    
    # JSON for full details (includes tracebacks as structured data)
    with open(output_dir / "batch_results.json", "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
```

### D. Proper Log Levels

| Level | Use Case | Goes To |
|-------|----------|---------|
| DEBUG | Detailed execution flow, variable values | Debug file only |
| INFO | Normal progress messages | Console + Debug file |
| WARNING | Something unexpected but handled | Console + Debug file |
| ERROR | Operation failed but continues | Console + Debug file |
| CRITICAL | Application cannot continue | Console + Debug file |

### E. Three Output Targets - Recommended Implementation

```python
from pathlib import Path
import logging
import sys
from dataclasses import dataclass
from typing import Literal
import json

@dataclass
class LoggingConfig:
    project_name: str
    diagnostics_dir: Path  # ./0_diagnostics/
    
    def setup(self) -> None:
        """Configure all logging targets once."""
        behavysis_logger = logging.getLogger("behavysis")
        behavysis_logger.setLevel(logging.DEBUG)
        
        # Prevent propagating to root (avoids double logging)
        behavysis_logger.propagate = False
        
        # Clear any existing handlers
        behavysis_logger.handlers = []
        
        # 1. Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(
            "%(levelname)s: %(message)s"
        ))
        behavysis_logger.addHandler(console)
        
        # 2. Debug file handler (~/.behavysis/{project}/debug.log)
        debug_dir = Path.home() / ".behavysis" / self.project_name
        debug_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(debug_dir / "debug.log", mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        ))
        behavysis_logger.addHandler(file_handler)
    
    def save_diagnostics(
        self,
        operation: str,
        results: list[dict]
    ) -> None:
        """Save structured diagnostics to ./0_diagnostics/."""
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        
        output = {
            "operation": operation,
            "results": results
        }
        
        with open(self.diagnostics_dir / f"{operation}.json", "w") as f:
            json.dump(output, f, indent=2)


# Usage in Project:
class Project:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.name = self.root_dir.name
        
        # Setup logging once
        self._logging_config = LoggingConfig(
            project_name=self.name,
            diagnostics_dir=self.root_dir / DIAGNOSTICS_DIR
        )
        self._logging_config.setup()
        
        self.logger = logging.getLogger(f"behavysis.project.{self.name}")
```

### F. Process Method Best Practice

```python
# Old way (current):
logger, io_obj = init_logger_io_obj()
logger.info("Doing work")
return get_io_obj_content(io_obj)

# New way (recommended):
import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)  # At module level

@dataclass
class ProcessOutput:
    status: Literal["success", "error", "skipped"]
    message: str
    # Additional structured data as fields
    records_processed: int = 0
    duration_seconds: float = 0.0

def my_process(experiment_name: str, ...) -> ProcessOutput:
    """Process returns structured data, not just log strings."""
    logger.info(f"Starting my_process for {experiment_name}")
    
    try:
        start_time = time.time()
        # ... do work ...
        duration = time.time() - start_time
        
        return ProcessOutput(
            status="success",
            message=f"Processed {records} records",
            records_processed=records,
            duration_seconds=duration
        )
    except Exception as e:
        logger.exception("Process failed")  # Logs full traceback at ERROR level
        return ProcessOutput(
            status="error",
            message=str(e)
        )
```

---

## Summary of Issues

| Issue | Severity | Problem | Solution |
|-------|----------|---------|----------|
| Shared debug.log | Medium | All projects write to same file | Project-specific debug files |
| Duplicate handlers | High | Each init adds new handlers | Configure once, use standard logger names |
| StringIO memory leak | Medium | Handlers never removed, StringIO grows | Use structured returns instead |
| Introspection for names | Low | `get_func_name_in_stack()` is fragile | Use `__name__` explicitly |
| Mixed concerns | High | Logging + diagnostics intertwined | Separate: console/logging vs structured diagnostics |
| Class-level loggers | Medium | All instances share logger | Instance loggers with project-specific names |

---

## Recommended Implementation Order

1. **Phase 1**: Replace `logging_utils.py` with standard logging config
2. **Phase 2**: Update `Project` and `Experiment` to configure logging once
3. **Phase 3**: Create `ProcessResult` dataclass for structured outputs
4. **Phase 4**: Update all process methods to return `ProcessResult` instead of strings
5. **Phase 5**: Update `_proc_scaff` to save structured diagnostics
6. **Phase 6**: Remove StringIO-based capture entirely
7. **Phase 7**: Test all three output targets work correctly
