"""Unit tests for behavioural classifier dataloaders and models."""

import numpy as np
import torch

from behavysis.behaviour_classifier.torch.base import TorchModel
from behavysis.behaviour_classifier.torch.dataset import WindowDataset


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
