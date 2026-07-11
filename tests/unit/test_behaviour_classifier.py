"""Unit tests for behavioural classifier dataloaders and models."""

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from behavysis.behaviour_classifier.adapter import SklearnAdapter, select_features
from behavysis.behaviour_classifier.config import TrainingRecipe
from behavysis.behaviour_classifier.torch.base import TorchModel
from behavysis.behaviour_classifier.torch.dataset import WindowDataset


def _recipe(**kw) -> TrainingRecipe:
    base = dict(
        model_type="logreg",
        behaviour_name="behav",
        individuals=["m1"],
        bodyparts=["nose"],
    )
    base.update(kw)
    return TrainingRecipe(**base)


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
    """Feature mask flows through fit/predict and joblib round-trip."""

    def test_fit_predict_shapes(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.standard_normal((60, 6))
        x[:, 1] = 0.0  # constant column dropped by selection
        y = rng.integers(0, 2, 60)
        adapter = SklearnAdapter(LogisticRegression(max_iter=200))
        adapter.fit([x], [y], [np.arange(60)], _recipe())
        assert 1 not in adapter.feature_mask
        prob = adapter.predict(x)
        assert prob.shape == (60,)

    def test_joblib_roundtrip(self, tmp_path) -> None:
        rng = np.random.default_rng(3)
        x = rng.standard_normal((60, 6))
        y = rng.integers(0, 2, 60)
        adapter = SklearnAdapter(LogisticRegression(max_iter=200))
        adapter.fit([x], [y], [np.arange(60)], _recipe())
        adapter.save(tmp_path)
        loaded = joblib.load(tmp_path / "model.joblib")
        np.testing.assert_array_equal(loaded.feature_mask, adapter.feature_mask)
        np.testing.assert_allclose(loaded.predict(x), adapter.predict(x))


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
        adapter.fit([x], [y], [np.arange(60)], _recipe(epochs=1, batch_size=16))
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
