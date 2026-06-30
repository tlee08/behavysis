# Simplify behavior classifier: start row-by-row

**Session ID:** ses_0e6bd9bf7ffejPooBMtRxe70Ky
**Created:** 7/1/2026, 2:00:09 AM
**Updated:** 7/1/2026, 2:18:46 AM

---

## User

Look at @src/behavysis/behav_classifier/ . I am concerned that this is overcomplicated. Let's start off with a simple row-by-row classifier (no CNN), and have separate functions to convert to windows for CNN (surely it's cleaner to load the window as it comes along and neglibible effect on compute, rather than pre-compute and load the window?).
Give your recommendations as a Principle Data Scientist. Be honest, be harsh but fair. Use your critical thinking and be inventive yet rigorous with your ideas. use tavily to inform your research and recommendations. Use karpathy guidelines. Can use context7

---

## Principal Data Scientist Review: `behaviour_classifier`

### Verdict: Overcomplicated. You have a ~1,000 line codebase to do what a 200-line sklearn pipeline could do.

---

### 1. Architecture Smell: PyTorch framework forced onto non-PyTorch models

`RF1` (Random Forest) inherits from `sklearn.ensemble.RandomForestClassifier`, but is called inside `BehaviourClassifier` which assumes `self.clf` is a `BaseTorchModel` (`clf_models/clf_templates.py:13-51`). `RF1` has to fake a `fit()` signature with `*args, **kwargs` and return a dummy empty DataFrame for training "history." **This is a type-system lie.** The code is pretending RF1 is a PyTorch model when it's not. The `BehaviourClassifier.clf` setter (`behaviour_classifier.py:117-131`) type-annotates as `BaseTorchModel | str` but silently accepts `RF1` (which is a `RandomForestClassifier`). This will blow up if any PyTorch-specific method is called on it.

**Recommendation:** Build a clean sklearn path first. A `RowByRowClassifier` that wraps any sklearn estimator. No DataLoader, no batches, no GPU. Fit on numpy, predict on numpy. ~50 lines.

---

### 2. The `listofvects2array` / `array2listofvect` antipattern

Used across `data.py:238-262` and `base_torch_model.py:57-68`. The code repeatedly:

1. Takes a list of per-video index/label arrays
2. Packs them into a flat `(video_id, row_id, label)` table
3. Does a stratified split or resampling on this table
4. Unpacks back into per-video lists

This is a manual, error-prone reimplementation of what `StratifiedGroupKFold` or a simple group-aware split function does natively. The intermediate representation has no semantics — column 0 is "video index", column 1 is "row index", column 2 is "label". Debugging this when shapes don't align is painful.

**Recommendation:** Delete both functions. Use `StratifiedGroupKFold` from sklearn, or write a 5-line group-stratified split. For the oversample/undersample: apply it on the concatenated training data directly, or per-video then concatenate. No need for the round-trip.

---

### 3. `MemoizedTimeSeriesDataset` is premature optimization (and wrong)

`MemoizedTimeSeriesDataset` (`base_torch_model.py:279-299`) caches every `__getitem__` call in a dict. Problems:

- **Training**: With `shuffle=True`, every epoch generates a new random permutation of indices. The cache never hits across epochs. It just fills memory.
- **Memory unbounded**: If you train on 100K samples × 21-frame windows, the dict grows to 100K entries of `(tensor, tensor)`. Each entry is ~40KB (498 features × 21 frames × 4 bytes). That's ~4GB of RAM for caching something that's just a numpy slice + transpose.
- **Produces wrong data on the second call if data mutates** (which it doesn't, but the pattern is fragile).
- The `__getitem__` slice-n-transpose is O(window_size) with no GPU transfer needed (that happens in the training loop). It's already fast.

**The user is right:** window-on-the-fly during `__getitem__` is the correct pattern. The padding is pre-computed (stored in `self.x_ls`), and the slice is O(1) in numpy. This is how PyTorch datasets are supposed to work. Delete `MemoizedTimeSeriesDataset`.

---

### 4. Magic numbers everywhere, untethered from data

| Location              | Magic Number    | What It Represents                                  |
| --------------------- | --------------- | --------------------------------------------------- |
| `data.py:118`         | `48`            | Columns to skip (2 indivs × 8 bodyparts × 3 coords) |
| `clf_templates.py:61` | `498`           | Features after column selection                     |
| `clf_templates.py:61` | Comment `# 546` | Original feature count?                             |
| `clf_templates.py:61` | `0`             | `window_frames=0` (meaning "no window")             |

These numbers must remain consistent across data loading, preprocessing, and model initialization. One column change in the feature extractor silently breaks the model with a tensor shape error at runtime. The `nfeatures` value (498) isn't read from the data — it's hardcoded in every model class.

**Recommendation:** Read `nfeatures` from the actual data shape. Assert it matches expected. Move `SELECTED_FEATURE_START = 48` to a named constant in `constants/`.

---

### 5. Feature selection via `FunctionTransformer` in a sklearn `Pipeline` adds indirection

`preproc_x_fit` builds a `Pipeline([FunctionTransformer(_select_derived_features), MinMaxScaler()])` and pickles it. But `_select_derived_features` is just `lambda x: x[:, 48:]`. Saving this as a pipeline step means:

- You can't inspect what columns were selected without loading the pipeline
- The pipeline file (`preproc.sav`) is fragile across code changes (if `_select_derived_features` changes, old pipelines break silently)
- It's different from the PyTorch path, which doesn't use this pipeline at all (the CNN/DNN models receive data pre-processed by `preproc_x_transform` but the feature selection is already baked in)

**Recommendation:** Drop the Pipeline. Use a standalone `MinMaxScaler` and a named constant for the column slice. Pickle only the scaler.

---

### 6. The training flow forces all data into memory and duplicates it

`prepare_training_data` (`data.py:175-261`):

- Loads all feature DataFrames → converts to list of numpy arrays
- Loads all label DataFrames → converts to numpy
- Concatenates everything to fit the scaler
- Applies scaler to each per-video array individually
- Flattens to index-table, splits, resamples, un-flattens
- Returns lists of full numpy arrays

For a pipeline that should scale, this loads everything into RAM. For 100 videos × 50K frames each × 500 features × 8 bytes = ~20GB. This should use memory-mapped files or a proper lazy dataset.

---

### 7. RF1 spec is suspicious

`RF1.__init__` sets `n_estimators=2000, max_depth=3` (`clf_templates.py:19-25`):

- 2,000 trees with max depth 3 is almost never the right balance. If depth 3 is needed to prevent overfitting, you don't need 2,000 trees — 200 would give essentially identical results. If 2,000 trees are needed for stability, you probably want deeper trees and fewer of them.
- `n_jobs=16` hardcoded assumes a 16-core machine.
- `verbose=1` in a pipeline will spam stdout.

---

### 8. The recommended refactor: what "simple" looks like

**Phase 1 — Row-by-row sklearn classifier (drop-in replacement for RF1, ~100 lines):**

```python
# row_classifier.py
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

class RowByRowClassifier:
    """Frame-by-frame classifier. No temporal context, no PyTorch, no DataLoader."""
    def __init__(self, estimator=None):
        self.estimator = estimator or RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        self.scaler = MinMaxScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = X[:, FEATURE_START:]  # constant from config
        X = self.scaler.fit_transform(X)
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = X[:, FEATURE_START:]
        X = self.scaler.transform(X)
        return self.estimator.predict_proba(X)[:, 1]

    def save(self, path): joblib.dump(self, path)

    @classmethod
    def load(cls, path): return joblib.load(path)
```

**Phase 2 — Window converter as a pure function (~30 lines):**

```python
# windows.py
def to_windows(X: np.ndarray, window_frames: int) -> np.ndarray:
    """Pad edges, return stacked windows of shape (n_samples, 2*window_frames+1, n_features)."""
    pad = np.pad(X, ((window_frames, window_frames), (0, 0)), mode='edge')
    return np.stack([pad[i:i + 2*window_frames+1] for i in range(X.shape[0])])
```

This is called once before training a temporal model. For large data, use `torch.utils.data.Dataset` (sliding on `__getitem__`, no memoization). The cost of slicing is negligible compared to the forward pass through a CNN.

**Phase 3 — Keep PyTorch models but decouple training from model definition:**

- `BaseTorchModel` becomes just `nn.Module` with `forward()`
- `fit_torch(model, train_dl, val_dl, epochs)` is a standalone function
- `predict_torch(model, x, window_frames)` is a standalone function
- Delete `MemoizedTimeSeriesDataset`, simplify `TimeSeriesDataset` to not require `y_ls` for prediction (use a dummy tensor internally, don't force the caller to create it)

**Phase 4 — Stratified train/test split without the index table:**

```python
from sklearn.model_selection import StratifiedGroupKFold

def stratified_split_by_video(x_ls, y_ls, test_size, random_state=42):
    X = np.concatenate(x_ls)
    y = np.concatenate(y_ls)
    groups = np.concatenate([np.full(len(x), i) for i, x in enumerate(x_ls)])
    sgkf = StratifiedGroupKFold(n_splits=int(1/test_size), shuffle=True, random_state=random_state)
    train_idx, test_idx = next(sgkf.split(X, y, groups))
    # Return indices per original video list...
```

---

### Bottom Line

| Issue                                    | Severity                             | Fix Priority |
| ---------------------------------------- | ------------------------------------ | ------------ |
| RF1 pretending to be PyTorch             | High — type error waiting to happen  | 1            |
| `listofvects2array` / `array2listofvect` | High — unnecessary complexity        | 1            |
| `MemoizedTimeSeriesDataset`              | Medium — memory leak, no benefit     | 2            |
| Hardcoded feature counts (498, 48)       | High — silent break on config change | 1            |
| Over-generic `BaseTorchModel`            | Medium — mixes concerns              | 2            |
| Full-data-in-memory training             | Medium — doesn't scale               | 3            |
| `FunctionTransformer` pipeline           | Low — unnecessary abstraction        | 3            |
| 2000 trees with depth 3                  | Low — suboptimal but not broken      | 3            |

**You were right to be concerned.** The simplest thing that works — a Scikit-learn random forest on per-frame features — is buried under layers of PyTorch abstractions, a custom index-packing scheme, and a memoized dataset that doesn't actually help. The literature confirms: RF on engineered DLC features is a strong, published baseline (SimBA uses exactly this). Start there, measure, and only then add temporal windows and deep models where the data demands it.

Shall I proceed with the refactor plan?
<system-reminder>
But seriously, if you need to edit or write to files, remember you're in read-only mode. Use the plan above and get confirmation from the user before making changes.</system-reminder>

---
