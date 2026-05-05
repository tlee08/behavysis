# Behavysis Architectural Recommendations

### 6. Plugin Architecture for Extensibility

#### Current Limitation

- Hard-coded processing functions
- Limited customization options

#### Recommended Plugin System

```python
class PluginManager:
    def __init__(self, plugin_dir: Path):
        self.plugin_dir = plugin_dir
        self.plugins: Dict[str, Type[ProcessingPlugin]] = {}
        self._discover_plugins()

    def register_plugin(self, name: str, plugin_class: Type[ProcessingPlugin]) -> None:
        self.plugins[name] = plugin_class

    def get_processor(self, name: str, config: Dict) -> ProcessingPlugin:
        return self.plugins[name](config)

@dataclass
class ProcessingPlugin(ABC):
    config: Dict

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        pass

# Example plugin implementation
class CustomFeatureExtractor(ProcessingPlugin):
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        # Custom feature extraction logic
        return enhanced_data

    def validate_config(self) -> bool:
        return "required_param" in self.config
```

### 7. Monitoring & Observability Architecture

#### Current Gap

- Limited logging
- No performance metrics
- No health checks

#### Recommended Monitoring Stack

```python
class MetricsCollector:
    def __init__(self):
        self.metrics_registry = CollectorRegistry()
        self._setup_metrics()

    def _setup_metrics(self):
        self.processing_duration = Histogram(
            'processing_duration_seconds',
            'Time spent processing experiments',
            ['pipeline_stage'],
            registry=self.metrics_registry
        )

        self.error_counter = Counter(
            'processing_errors_total',
            'Total processing errors',
            ['error_type', 'pipeline_stage'],
            registry=self.metrics_registry
        )

    @contextmanager
    def track_processing(self, stage: str):
        start_time = time.time()
        try:
            yield
        except Exception as e:
            self.error_counter.labels(error_type=type(e).__name__, stage=stage).inc()
            raise
        finally:
            duration = time.time() - start_time
            self.processing_duration.labels(stage=stage).observe(duration)

# Usage
with metrics.track_processing("dlc_inference"):
    experiment.run_dlc()
```

### Improve `df_classes`

Now I have a clear picture. Let me provide my assessment using Karpathy's guidelines.

---

## Assessment

**My assumptions:**

1. You want validation to catch bugs early in the pipeline
2. You're the sole maintainer, so consistency matters more than public API

**Verdict: It's overcomplicated in places, but the core pattern is sound.**

---

### What's Good

The basic mixin pattern is reasonable:

```python
class KeypointsDf(DFMixin):
    IN = FramesIN
    CN = KeypointsCN
```

This gives you schema documentation in code and catches index/column name mismatches.

---

### What's Overcomplicated

**1. `check_df` is never called directly**

The validation only runs inside `read()` → `basic_clean()` → `check_df()`. But `check_df()` is never called on `write()`. So you validate on input but not output—meaning corrupted data could still be written.

**Recommendation:** Either call `check_df()` in `write()`, or remove `check_df` entirely and just do inline assertions where you actually need them.

**2. The inheritance chain adds complexity for little gain**

`BehavDf` → `BehavPredictedDf` / `BehavScoredDf` with `OutcomesCols` class attributes. This is 75 lines to validate that a DataFrame has `(behav, prob)` or `(behav, pred)` columns.

Compare to inline:

```python
assert ("behav", "prob") in df.columns, f"Missing prob column for {behav}"
```

**3. `enum2tuple` helper**

From `misc_utils.py`—this converts enums to tuples. But you could just do:

```python
[c.value for c in MyEnum]
```

instead of importing a helper.

**4. `basic_clean` does too much**

It sets index names, sets column names, sorts, AND validates. These should be separate concerns. Sorting especially is a side effect that could cause performance issues on large dataframes.

---

#### What Follows Best Principles

The pattern of having typed schemas (IN/CN enums) is good—it documents what the dataframe should look like and catches bugs at runtime.

But ask yourself: **"Would a senior engineer say this is overcomplicated?"**

For validating pandas schemas, probably yes. A simpler approach:

```python
# Just use plain functions, no classes needed
def validate_keypoints(df: pd.DataFrame) -> None:
    assert df.index.name == "frame"
    assert df.columns.names == ["individuals", "bodyparts", "coords"]
    assert not df.isna().any().any()
```

No mixin, no inheritance, no `enum2tuple`. Same validation, 90% less code.

---

#### My Recommendation

If you want to keep the pattern, simplify:

1. **Remove the class inheritance** for dataframe validators—just have one `DFMixin` and use composition instead
2. **Remove `check_df` as a separate method**—inline the assertions in `read/write` where they're needed
3. **Remove sorting from `basic_clean`**—sort only when you need to, not on every read
4. **Consider whether you need runtime validation at all**—if your pipeline produces consistent data, unit tests + type hints might be enough

If you're finding actual bugs that these validators catch, keep them. If they're just ceremony you work around, simplify.
