"""Unit tests for behavioural classifier dataloaders and models."""

import joblib
import numpy as np
import polars as pl
import torch
from sklearn.linear_model import LogisticRegression

from behavysis.behaviour_classifier.adapter import SklearnAdapter, select_features
from behavysis.behaviour_classifier.config import TrainingRecipe
from behavysis.behaviour_classifier.torch.base import TorchModel
from behavysis.behaviour_classifier.torch.dataset import WindowDataset


def _recipe(**kw) -> TrainingRecipe:
    base = dict(model_type="logreg")
    base.update(kw)
    return TrainingRecipe(**base)


def _df(x: np.ndarray, y: np.ndarray, name: str = "test") -> pl.DataFrame:
    """Build a training DataFrame from numpy arrays."""
    cols = [f"feat_{i}" for i in range(x.shape[1])]
    return pl.DataFrame(x, schema=cols).with_columns(
        pl.lit(name).alias("experiment"),
        pl.int_range(x.shape[0]).alias("frame"),
        pl.Series("actual", y),
    )


def _mask(df: pl.DataFrame) -> np.ndarray:
    """Full train mask (all rows in training)."""
    return np.ones(len(df), dtype=bool)


class TestFeatureSelection:
    """Tests for supervised feature selection."""

    def test_drops_constant_columns(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((50, 4))
        x[:, 2] = 1.0  # constant → zero variance
        y = rng.integers(0, 2, 50)
        keep = select_features(x, y, _recipe())
        assert 2 not in keep
        assert set(keep) == {0, 1, 3}

    def test_disabled_keeps_all(self) -> None:
        x = np.zeros((10, 5))
        y = np.zeros(10, dtype=int)
        keep = select_features(x, y, _recipe(feature_selection=False))
        assert list(keep) == [0, 1, 2, 3, 4]

    def test_max_features_caps(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.standard_normal((80, 10))
        y = rng.integers(0, 2, 80)
        keep = select_features(x, y, _recipe(max_features=3))
        assert len(keep) == 3
        assert list(keep) == sorted(keep)


class TestSklearnAdapterMask:
    """Pipeline flows through fit/predict and joblib round-trip."""

    @staticmethod
    def _pipeline(config):
        from imblearn.pipeline import Pipeline as ImbPipeline
        from sklearn.feature_selection import VarianceThreshold

        steps = []
        if config.feature_selection:
            steps.append(("var_filter", VarianceThreshold()))
        steps.append(("clf", LogisticRegression()))
        return ImbPipeline(steps)

    def test_fit_predict_shapes(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.standard_normal((60, 6))
        x[:, 1] = 0.0  # constant column dropped by variance filter
        y = rng.integers(0, 2, 60)
        adapter = SklearnAdapter(self._pipeline)
        df = _df(x, y)
        adapter.fit(df, _mask(df), _recipe())
        support = adapter.pipe.named_steps["var_filter"].get_support()
        assert not support[1]  # constant column dropped
        prob = adapter.predict(x)
        assert prob.shape == (60,)

    def test_joblib_roundtrip(self, tmp_path) -> None:
        rng = np.random.default_rng(3)
        x = rng.standard_normal((60, 6))
        y = rng.integers(0, 2, 60)
        adapter = SklearnAdapter(self._pipeline)
        df = _df(x, y)
        adapter.fit(df, _mask(df), _recipe())
        adapter.save(tmp_path)
        loaded = joblib.load(tmp_path / "model.joblib")
        np.testing.assert_allclose(
            loaded.predict_proba(x)[:, 1],
            adapter.predict(x),
        )


class TestSklearnAdapterGridSearch:
    """All hyperparameters are lists — single values are single-element lists."""

    @staticmethod
    def _pipeline(config):
        from imblearn.pipeline import Pipeline as ImbPipeline
        from sklearn.ensemble import RandomForestClassifier as RFC

        return ImbPipeline([("clf", RFC())])

    @staticmethod
    def _logreg_pipeline(config):
        from imblearn.pipeline import Pipeline as ImbPipeline
        return ImbPipeline([("clf", LogisticRegression())])

    def test_grid_search_resolves_params(self) -> None:
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.default_rng(5)
        x = rng.standard_normal((80, 5))
        y = rng.integers(0, 2, 80)
        adapter = SklearnAdapter(self._pipeline)
        recipe = _recipe(
            feature_selection=False,
            hyperparameters={
                "clf__n_estimators": [10, 20],
                "clf__max_depth": [4, 8],
                "clf__random_state": [42],
            },
        )
        df = _df(x, y)
        adapter.fit(df, _mask(df), recipe)

        assert adapter.resolved_hyperparameters is not None
        assert "clf__n_estimators" in adapter.resolved_hyperparameters
        assert isinstance(adapter.resolved_hyperparameters["clf__n_estimators"], int)
        assert adapter.resolved_hyperparameters["clf__random_state"] == 42

    def test_single_option_lists_still_grid(self) -> None:
        rng = np.random.default_rng(6)
        x = rng.standard_normal((40, 3))
        y = rng.integers(0, 2, 40)
        adapter = SklearnAdapter(self._logreg_pipeline)
        recipe = _recipe(
            feature_selection=False,
            hyperparameters={
                "clf__C": [1.0],
                "clf__max_iter": [500],
                "clf__random_state": [99],
            },
        )
        df = _df(x, y)
        adapter.fit(df, _mask(df), recipe)

        rhp = adapter.resolved_hyperparameters
        assert rhp is not None
        assert rhp["clf__C"] == 1.0
        assert rhp["clf__random_state"] == 99
        assert hasattr(adapter.pipe_, "best_params_")
        prob = adapter.predict(x)
        assert prob.shape == (40,)



class TestTorchAdapterMask:
    """Torch adapter save/load reconstructs nfeatures from the mask."""

    def test_save_load_roundtrip(self, tmp_path) -> None:
        from behavysis.behaviour_classifier.adapter import TorchAdapter
        from behavysis.behaviour_classifier.torch.architectures import DNN1

        rng = np.random.default_rng(4)
        x = rng.standard_normal((60, 8)).astype(np.float32)
        x[:, 3] = 0.0  # constant → dropped
        y = rng.integers(0, 2, 60).astype(np.float32)
        adapter = TorchAdapter(lambda nf: DNN1(nf, window_frames=0))
        df = _df(x, y)
        adapter.fit(df, _mask(df), _recipe(epochs=1, batch_size=16))
        assert 3 not in adapter.feature_mask
        adapter.save(tmp_path)

        reloaded = TorchAdapter(lambda nf: DNN1(nf, window_frames=0))
        reloaded.load_state(tmp_path)
        assert reloaded.model.nfeatures == len(adapter.feature_mask)
        np.testing.assert_allclose(reloaded.predict(x), adapter.predict(x), atol=1e-5)


def make_x_y_arrays(nrows: int, ncols: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate random feature and label arrays."""
    x = np.random.randn(nrows, ncols).astype(np.float32)
    y = np.random.randint(0, 2, size=(nrows,)).astype(np.float32)
    return x, y


class TestWindowDataset:
    """Tests for WindowDataset (sliding windows, no memoization)."""

    def test_training_dataloader(self) -> None:
        """WindowDataset should produce correct windowed samples."""
        nrows = 100
        ncols = 500
        nind = 10
        window_frames = 5

        x, y = make_x_y_arrays(nrows, ncols)
        index = np.random.choice(nrows, nind, replace=False)

        ds = WindowDataset(
            x_ls=[x],
            y_ls=[y.reshape(-1)],
            index_ls=[index],
            window_frames=window_frames,
        )

        assert len(ds) == nind
        sample_x, _sample_y = ds[0]
        assert sample_x.shape == (ncols, 2 * window_frames + 1)
        assert sample_x.dtype == torch.float32

    def test_inference_dataloader(self) -> None:
        """WindowDataset works for inference with dummy labels."""
        nrows = 100
        ncols = 500
        window_frames = 5
        nind = 10

        x, _ = make_x_y_arrays(nrows, ncols)
        y = np.zeros(nrows, dtype=np.float32)
        index = np.random.choice(nrows, nind, replace=False)

        ds = WindowDataset(
            x_ls=[x],
            y_ls=[y],
            index_ls=[index],
            window_frames=window_frames,
        )

        assert len(ds) == nind
        sample_x, sample_y = ds[0]
        assert sample_x.shape == (ncols, 2 * window_frames + 1)
        assert sample_y.shape == (1,)

    def test_zero_window(self) -> None:
        """WindowDataset with window_frames=0 returns 1-frame windows."""
        nrows = 10
        ncols = 500

        x, y = make_x_y_arrays(nrows, ncols)
        index = np.arange(nrows)

        ds = WindowDataset(
            x_ls=[x],
            y_ls=[y],
            index_ls=[index],
            window_frames=0,
        )

        assert len(ds) == nrows
        sample_x, _ = ds[0]
        assert sample_x.shape == (ncols, 1)


class TestTorchModel:
    """Tests for TorchModel base class."""

    def test_model_initialization(self) -> None:
        """Model should initialize with correct dimensions."""
        nfeatures = 100
        window_frames = 10

        class DummyModel(TorchModel):
            def __init__(self, nf, wf):
                super().__init__(nf, wf)
                self.linear = torch.nn.Linear(nf, 1)

            @property
            def criterion(self):
                return torch.nn.BCELoss()

            @property
            def optimizer(self):
                return torch.optim.Adam(self.parameters())

            def forward(self, x):
                return torch.sigmoid(self.linear(x[:, :, 0]))

        model = DummyModel(nfeatures, window_frames)
        assert model.nfeatures == nfeatures
        assert model.window_frames == window_frames

    def test_device_property_cpu(self) -> None:
        """Device property should work correctly for CPU."""

        class DummyModel(TorchModel):
            def __init__(self, nfeatures, window_frames):
                super().__init__(nfeatures, window_frames)
                self.linear = torch.nn.Linear(nfeatures, 1)

            @property
            def criterion(self):
                return torch.nn.BCELoss()

            @property
            def optimizer(self):
                return torch.optim.Adam(self.parameters())

            def forward(self, x):
                return torch.sigmoid(self.linear(x[:, :, 0]))

        model = DummyModel(100, 10)
        model.to_device(torch.device("cpu"))
        assert model.device.type == "cpu"
