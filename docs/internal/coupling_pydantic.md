# Paradigm viability assessment and config coupling

**Session ID:** ses_0e4b903a1ffeXlgZNoMz51UQ8B
**Created:** 7/1/2026, 11:24:24 AM
**Updated:** 7/1/2026, 11:28:22 AM

---

## User

You are a principle data scientist and data engineer.
Be critical, verify what you do, be elegant in your solutions, be honest and harsh but fair.

Context: I see that there's a lot of coupling between individual functions in @src/behavysis/funcs/ and the config Pydantic models in @src/behavysis/models/funcs/ (e.g. for calculate_parameters, preprocess, and analyse).

Your task: Critically assess the viability of this current paradigm. Research other paradigms that are a better fit. Be effective, be elegant, be creative. Present your assessment.

Aside: Also investigate the current structure of the configs Pydantic models. Are they overcomplicated? Users have commented that they've found knowing what values to set and troubleshooting why configs are wrong to be confusing.

---

## Critical Assessment: Config-Function Coupling in Behavysis

### 1. The Core Architecture Pattern

The pipeline follows a **file-based config-as-state** pattern:

```
┌─────────────────────────────────────────────────────────────┐
│  config.json  (ExperimentConfig)                            │
│  ├── user:     UserConfig     (user-facing params)          │
│  ├── auto:     AutoConfig     (computed, mutated by funcs)  │
│  └── ref:      RefConfig      (extensible key-value store)  │
└──────────────┬──────────────────────────────────────────────┘
               │  Every function calls:
               │  config = ExperimentConfig.model_validate_json(
               │      config_fp.read_text())
               │
    ┌──────────┴──────────┬──────────────────┐
    ▼                     ▼                   ▼
 format_video()     calculate_parameters()    analyse.in_roi()
 (reads user.*)     (reads user.*,        (reads user.*,
 (writes auto.*)     writes auto.*)        reads auto.*)
```

Every function reads the **entire** `ExperimentConfig` from disk, extracts the few fields it needs, and in some cases **mutates** and writes it back. The functions and configs are mirror-image directories: `models/funcs/analyse.py` defines the config shapes that `funcs/analyse.py` consumes.

### 2. Critical Problems

**A. Config-file-as-database: fragile, non-atomic, untestable**

The worst anti-pattern is in `calculate_parameters.py`. Functions there:

1. Parse config from disk
2. Compute a value
3. **Re-parse config from disk** (defensive against concurrent writes)
4. Mutate `config.auto.*`
5. Write it back

This is a hand-rolled file-based state machine with no transactional guarantees. If two calculate-param functions run in parallel, the second write wins. The pattern leaks the config file into being **both user input and pipeline internal state** — users can't regenerate `auto` fields without running the full pipeline. This makes config files illegible after a pipeline run (they're filled with mysterious `auto.*` values like `start_frame: -1` vs `start_frame: 14530`).

**Verdict: Broken.** Config should be immutable input. Computed metadata belongs in a separate artifact store or database.

**B. Missing validation: users are flying blind**

This is your users' actual complaint. Across all 25 Pydantic model classes, there are **zero** `@field_validator`, `@model_validator`, or `@computed_field` decorators. Zero `Field(gt=0, le=1)` constraints. The only runtime validation is four `assert` statements in `get_analysis_config()` (line 108-111 of `experiment_config.py`).

Practical consequences:

- A user sets `fps: -5` → no error on config load → fails with cryptic `AssertionError` 3 pipeline steps later in `extract_features`
- A user sets `pcutoff: 500` → silently accepted → produces nonsensical likelihood thresholds
- A user types `bodyparts: ["LeftEar", "RightEar"]` → valid config → but `"Nose"` is missing and subtle results errors appear downstream
- The `float | str` union type accepts `"--typo_in_ref_name"` → passes Pydantic validation → fails at runtime with `AssertionError: Value 'typo_in_ref_name' can't be found`

**Verdict: User confusion is entirely justified.** The config system provides zero guardrails. Pydantic's validation engine is present but completely unused.

**C. The `get_ref()` indirection mechanism is clever but opaque**

The idea is elegant: define shared values in `ref`, reference them as `--bpts_simba`. But:

1. `RefConfig` uses `extra="allow"` — it's an untyped grab-bag. There's no way to discover what reference names exist.
2. The `float | str` type means Pydantic treats `"--bpts_simba"` as a valid float (it isn't; it passes because `str` is accepted). The discriminant is purely a string prefix convention, not a discriminated union.
3. `get_ref()` is called in **12 separate locations** across the codebase, each with the same pattern. It's a de facto cross-cutting concern with no abstraction.

**D. Deep coupling through deep nesting**

The chain `config.user.analyse.in_roi[0].bodyparts` is 5 levels deep. Each func accesses config at that depth through hard-coded attribute chains. If you rename `analyse` to `analysis_measures` in the model, you must update every function that touches it — no type-safe refactoring path.

**E. Protocol classes are decorative**

`AnalyseFunc`, `CalculateParamsFunc`, and `PreprocessFunc` are defined but:

- Never used in `isinstance` checks
- Never used for registration/dispatch
- Not checked at module boundaries (functions could violate the protocol silently)
- Only referenced as type hints in `Experiment.calculate_parameters(funcs: tuple[CalculateParamsFunc, ...])` — but the actual call is `func(keypoints_fp=..., config_fp=...)` with **keyword arguments**, which Protocols don't validate

**F. Dead code: `simba_features.py`**

This file duplicates the SimBA computation from `extract_features.py` but is never imported or re-exported. It's noise that confuses the architecture.

### 3. Viability Verdict

The current paradigm works — the pipeline runs end to end — but it scores poorly on the software engineering axes that matter for a research codebase with multiple users:

| Dimension       | Current                      | Target                            |
| --------------- | ---------------------------- | --------------------------------- |
| Config error UX | Fail late, cryptic           | Fail fast, descriptive            |
| Testability     | Must mock file I/O           | Pure functions, in-memory         |
| Refactorability | String-coupled to JSON paths | Type-safe IDE refactoring         |
| Discoverability | No field docs/constraints    | Self-documenting config           |
| Parallel safety | File-based, racy             | Immutable config                  |
| Onboarding time | Days of trial-and-error      | Config works or tells you why not |

### 4. Recommended Paradigm

I recommend a **stratified config architecture** with three layers:

```
┌──────────────────────────────────────────────┐
│ LAYER 1: Pipeline step descriptors           │
│ (dataclass/Pydantic — validated, immutable)  │
│                                              │
│ Each pipeline step declares:                 │
│   - input schema / output schema             │
│   - required config subsection               │
│   - the function to invoke                   │
└──────────────┬───────────────────────────────┘
               │
┌──────────────▼───────────────────────────────┐
│ LAYER 2: Domain-specific configs             │
│ (Pydantic + FULL validators)                 │
│                                              │
│ Each config is narrow:                       │
│   - FormatVidConfig: validated params        │
│   - PxPerMmConfig: validated params          │
│   - Not nested under a monolithic root       │
│   - References resolved AT LOAD TIME         │
└──────────────┬───────────────────────────────┘
               │
┌──────────────▼───────────────────────────────┐
│ LAYER 3: Computed metadata (separate store)  │
│ (SQLite / parquet / key-value file)          │
│                                              │
│ "auto" fields should NEVER be in config.     │
│ Steps produce artifacts that downstream      │
│ steps consume. Config stays immutable.       │
└──────────────────────────────────────────────┘
```

**Concrete changes:**

1. **Add Pydantic validators everywhere.** `Field(gt=0, le=1)` on `pcutoff`. `@field_validator` on `bodyparts` to check against valid bodypart names. `@model_validator` for cross-field constraints. This alone solves 80% of the user confusion.

2. **Resolve references at construction time, not lazily.** Instead of `float | str` with runtime `get_ref()`, use a config loader that resolves `RefConfig` references when building the config tree. The runtime models have clean types: `bodyparts: list[str]`, `pcutoff: float`, never `str`.

3. **Separate config from metadata.** Move `auto.*` out of `ExperimentConfig` into a separate stage-output store. Each step's output is an artifact file; the orchestrator tracks what's been computed. The `-1` sentinel values for uncomputed fields vanish.

4. **Inject configs, don't read them from disk in functions.** Functions should receive the specific config object they need (e.g., `InRoiConfig`), not a file path to a monolithic JSON. The `Experiment`/`Project` orchestration layer handles I/O; functions are pure(ish) transforms.

5. **Actually use the Protocols.** Define a registry where each pipeline step registers itself with its Protocol, config type, and I/O shape. The pipeline runner can then validate config completeness and execution order statically.

6. **Delete `simba_features.py`** or merge it into `extract_features.py`.

### 5. Bottom Line

The current paradigm is **functional but fragile**. It's a prototype that grew organically. The coupling is deep, the validation is absent, and the config mutation pattern will cause increasingly subtle bugs as the user base grows. This isn't a rewrite — it's a refactor with a clear payoff. The highest-leverage single change is adding Pydantic validators to all config models; the second is resolving references at load time rather than lazily. Together they eliminate the user-facing confusion directly.

---

## User

Ok, this is what I'm planning. Give your assessment of the following plan:

1. Split config into config and metadata (already updated @src/behavysis/constants/pipeline.py ). Config is now immutable. Metadata now for writing auto-computed vals.
2. Convert config (and metadata) to yaml instead of json. This means we can use yaml's inbuilt variable assignment rather than the opaque get_ref.
3. Figure out a way to structure the config.yaml file nicely so that a) only configs that we need can be included (e.g. can be an almost empty file), and is extensible such that it can take all the needed configs, and there's some kind of taxonomy to config nesting.
4. Eventually add validation for each model.

Note:

- Don't worry about protocol classes only being type checking for now. This is fine.
- simba_features.py removed now
- Where should we move the mirror models/funcs/ modules to? or should we keep in the current place?

---

## Assessment of Your Plan

### Point 1: Split config ↔ metadata

**Strong yes.** This was the #1 architectural flaw. The `get_fp(METADATA_DIR)` addition in `pipeline.py` is clean. One design consideration: currently `AutoConfig` uses `-1` as sentinel for "not yet computed." With a separate metadata file, you can simply omit uncomputed fields — `Optional` types or just absent keys. Cleaner semantics, no magic values.

### Point 2: YAML with native anchors replacing `get_ref`

**Good direction, one non-trivial caveat.** YAML anchors (`&name` / `*name`) resolve the entire indirection problem at parse time. The `float | str` and `list[str] | str` union types collapse to clean `float` and `list[str]`. The `get_ref()` method and the entire `RefConfig` class disappear. Types become honest.

The caveat: **Anchor definitions need a home in the YAML document.** You can't define `&bpts_simba` floating in space — it must be a value under some key. There are three clean options:

| Option                                             | How it looks                                                           | Trade-off                                                             |
| -------------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------- |
| **A. `definitions:` section + `extra="ignore"`**   | Top-level `definitions:` key holds all anchors; Pydantic drops it      | Cleanest. One known key that's always ignored.                        |
| **B. Inline at first use**                         | `bodyparts: &bpts_simba [LeftEar, Nose, ...]` at each first occurrence | No dummy section needed, but anchors are scattered and harder to find |
| **C. `extra="allow"` with unstructured top-level** | Anchors live as top-level keys like `bpts_simba: &bpts_simba [...]`    | Pydantic silently accepts them; they pollute the model's namespace    |

**Recommend A.** Single well-known location, parse-and-forget. `ExperimentConfig` gets `model_config = ConfigDict(extra="ignore")`.

One more thing: PyYAML aliases create **the same Python object in memory** for all references to a `*name`. Since config is now immutable, this is fine — but any code that accidentally mutates a resolved list will corrupt all other references. Worth a comment in the codebase warning about this.

### Point 3: Minimal + extensible YAML structure

You already have this with Pydantic defaults. Omit any field and it fills from the model default. The real UX upgrade is **YAML comments** — JSON can't have them. A well-commented `default_config.yaml` serves as both documentation and template:

```yaml
# config.yaml — Behavysis experiment configuration
# Omit any field to use defaults. See default_config.yaml for all options.

user:
  format_vid:
    fps: 30 # Output frame rate. Omit to keep source fps.

  analyse:
    speed:
      smoothing_sec: 1.0 # Window for smoothing velocity. Higher = smoother.
      # bodyparts omitted → defaults to BPTS_SIMBA
```

The taxonomy/nesting question: the current 5-level max depth (`user.analyse.in_roi[0].bodyparts`) is actually reasonable for YAML. Deep nesting is a problem in code (brittle attribute chains), but in YAML it's visually scannable. Don't restructure the nesting — just make the leaves well-named and well-commented.

### Point 4: Validation

**This should NOT be deferred.** It's the highest-ROI change in your entire plan and directly addresses the user complaint. Adding validators now while restructuring is nearly free; adding them later means touching 25 model classes twice instead of once. Minimum viable validators:

- `@field_validator` on every `pcutoff`, `smoothing_sec`, `window_sec` → `Field(gt=0)`
- `pcutoff` specifically → `Field(ge=0, le=1)`
- `fps`, `width_px`, `height_px` → `Field(gt=0)`
- `bodyparts` → validator that checks against known bodypart names
- `metric` (the one `Literal`) → already self-validating, good

---

## Where to Put the Models

Options:

| Option                      | Structure                                                        | Pros                                     | Cons                                                                                    |
| --------------------------- | ---------------------------------------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------- |
| **A. Colocate**             | `funcs/analyse.py` contains both `in_roi()` and `InRoiConfig`    | Single source of truth; one file to edit | Functions import `cv2`, `numpy`, `matplotlib` — importing config types pulls heavy deps |
| **B. Flatten to `models/`** | `models/analyse.py`, `models/preprocess.py` (no `funcs/` subdir) | Clean separation; lightweight imports    | Still two places to touch per change                                                    |
| **C. Per-stage package**    | `pipeline/analyse/config.py` + `pipeline/analyse/func.py`        | True domain grouping                     | Overkill for single-file modules                                                        |

**Recommend B.** Kill the `models/funcs/` subdirectory, flatten config modules directly into `models/`. The mirror structure (`models/funcs/analyse.py` ↔ `funcs/analyse.py`) adds cognitive overhead for zero benefit. The resulting `models/`:

```
models/
  __init__.py
  experiment.py         # ExperimentConfig, UserConfig, AnalysisConfig
  metadata.py           # MetadataConfig (was AutoConfig)
  format_vid.py         # FormatVidConfig, VidMetadata
  run_dlc.py            # RunDlcConfig
  calculate_parameters.py   # CalculateParamsConfig, FromLikelihoodConfig, etc.
  preprocess.py         # PreprocessConfig, InterpolateConfig, etc.
  extract_features.py   # ExtractFeaturesConfig
  classify_behaviour.py # ClassifyBehaviourConfig
  analyse.py            # AnalyseConfig, SpeedConfig, InRoiConfig, etc.
  bouts.py              # Bout, BoutStruct, Bouts
  classifier.py         # BehaviourClassifierConfig (was behaviour_classifier_config.py)
  examples/
    __init__.py         # get_default_config()
```

The import changes are mechanical: `from behavysis.models.funcs.analyse import AnalyseConfig` → `from behavysis.models.analyse import AnalyseConfig`.

---

## Additional Things Worth Addressing

**1. YAML I/O utility.** Pydantic has no built-in YAML support. You'll need a small wrapper, ideally replacing `model_validate_json(config_fp.read_text())` with something like:

```python
# In some utility or on ExperimentConfig itself
@classmethod
def from_yaml(cls, path: Path) -> "ExperimentConfig":
    import yaml
    return cls.model_validate(yaml.safe_load(path.read_text()))
```

This is ~10 lines and gets called 12+ times across the codebase. Worth centralizing.

**2. `update_config` function.** Currently handles JSON + `ExperimentConfig`. Needs updating for YAML + config/metadata split. Make it only touch `user` (metadata is managed by the pipeline, never user-edited). The `overwrite: Literal["user", "all"]` parameter collapses to just overwriting `user` — there is no `all` to overwrite anymore.

**3. Migration path.** Users will have existing JSON configs. A one-time migration script (`json → yaml`) is trivial: read old JSON, write new YAML. Worth bundling into the release.

**4. `get_analysis_config()`.** Currently bridges `auto.*` → validated `AnalysisConfig`. After the split, it bridges `metadata.*` → `AnalysisConfig`. The method should move to a utility that takes config + metadata as arguments, not live on `ExperimentConfig`.

---

## Verdict on the Plan

The direction is right. The priority order should be:

1. Split config/metadata + YAML + flatten models directory (all one mechanical refactor)
2. Add validators (same PR, every model file you touch)
3. Replace `get_ref()` with YAML anchors (falls out naturally from #1)
4. Comments in `default_config.yaml` for discoverability

The only risk: YAML anchors are a slightly obscure feature. Make sure `default_config.yaml` demonstrates them clearly so users can copy the pattern.

---
