"""Unit tests for behavioral classifier dataloaders."""

import numpy as np
import pytest

from behavysis.behav_classifier.clf_models.base_torch_model import (
    BaseTorchModel,
    MemoizedTimeSeriesDataset,
    TimeSeriesDataset,
)


def make_x_y_arrays(nrows: int, ncols: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate random feature and label arrays.

    Parameters
    ----------
    nrows : int
        Number of samples.
    ncols : int
        Number of features per sample.

    Returns:
    -------
    tuple[np.ndarray, np.ndarray]
        Feature array (nrows, ncols) and label array (nrows, 1).
    """
    x = np.random.randn(nrows, ncols)
    y = np.random.randn(nrows, 1)
    return x, y


class TestMemoizedTimeSeriesDataset:
    """Tests for MemoizedTimeSeriesDataset (training dataloader)."""

    def test_training_dataloaders(self) -> None:
        """Training dataloader should produce correct windowed samples."""
        # Parameters
        nrows = 100
        ncols = 546
        nind = 10
        window_frames = 5

        # Making x, y, index
        x, y = make_x_y_arrays(nrows, ncols)
        index = np.random.choice(nrows, nind, replace=False)

        # MemoizedTimeSeriesDataset expects lists
        train_ds = MemoizedTimeSeriesDataset(
            x_ls=[x], y_ls=[y], index_ls=[index], window_frames=window_frames
        )

        # Verify the dataset length matches
        assert len(train_ds) == nind

        # Verify window shape
        sample_x, _sample_y = train_ds[0]
        assert sample_x.shape == (ncols, 2 * window_frames + 1)

    def test_with_base_torch_model_fit_loader(self) -> None:
        """BaseTorchModel.fit_loader should produce correct windowed dataset."""
        nrows = 100
        ncols = 546
        nind = 10
        window_frames = 5

        x, y = make_x_y_arrays(nrows, ncols)
        index = np.random.choice(nrows, nind, replace=False)

        # BaseTorchModel.fit_loader expects lists
        clf = BaseTorchModel(ncols, window_frames)
        train_dl = clf.fit_loader([x], [y], [index], 32)
        train_ds = train_dl.dataset

        # Verify dataset length
        assert len(train_ds) == nind


class TestTimeSeriesDataset:
    """Tests for TimeSeriesDataset (inference dataloader)."""

    def test_inference_dataloaders(self) -> None:
        """Inference dataloader should produce correct windowed samples."""
        # Parameters
        nrows = 100
        ncols = 546
        window_frames = 5
        nind = 10

        # Making x, y, index
        x, y = make_x_y_arrays(nrows, ncols)
        y = np.zeros((nrows, 1))
        index = np.random.choice(nrows, nind, replace=False)

        # TimeSeriesDataset expects lists
        train_ds = TimeSeriesDataset(
            x_ls=[x], y_ls=[y], index_ls=[index], window_frames=window_frames
        )

        # Verify the dataset length matches
        assert len(train_ds) == nind

        # Verify window shape
        sample_x, _sample_y = train_ds[0]
        assert sample_x.shape == (ncols, 2 * window_frames + 1)

    def test_with_base_torch_model_predict_loader(self) -> None:
        """BaseTorchModel.predict_loader should produce correct windowed dataset."""
        nrows = 100
        ncols = 546
        window_frames = 5
        nind = 10

        x, _y = make_x_y_arrays(nrows, ncols)
        index = np.random.choice(nrows, nind, replace=False)

        # Test with BaseTorchModel's predict_loader
        clf = BaseTorchModel(ncols, window_frames)
        test_dl = clf.predict_loader(x, index, 32)
        train_ds = test_dl.dataset

        # Verify dataset length
        assert len(train_ds) == nind


class TestBaseTorchModel:
    """Tests for BaseTorchModel class."""

    def test_model_initialization(self) -> None:
        """Model should initialize with correct dimensions."""
        nfeatures = 100
        window_frames = 10
        model = BaseTorchModel(nfeatures, window_frames)

        assert model.nfeatures == nfeatures
        assert model.window_frames == window_frames

    def test_device_property_cpu(self) -> None:
        """Device property should work correctly for CPU."""
        model = BaseTorchModel(100, 10)
        # Set device to cpu - this should work
        import torch

        model._device = torch.device("cpu")
        model.cpu()

        assert model.device.type == "cpu"
