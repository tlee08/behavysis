import numpy as np
import torch
from torch.utils.data import TensorDataset

from behavysis_pipeline.behav_classifier.base_torch_model import (
    BaseTorchModel,
    MemoizedTimeSeriesDataset,
    TimeSeriesDataset,
)


def make_x_y_dfs(nrows, ncols):
    x = np.random.randn(nrows, ncols)
    y = np.random.randn(nrows, 1)

    return x, y


def check_windows(ds_windows, ds_orig, window_frames):
    for i, (a, b) in enumerate(zip(ds_windows, ds_orig)):
        assert np.all(np.isclose(a[0][:, window_frames], b[0].numpy()))
        assert np.all(np.isclose(a[1], b[1].numpy()))


def test_training_dataloaders():
    # Parameters
    nrows = 100
    ncols = 546
    nind = 10
    window_frames = 5

    # Making x, y, index
    x, y = make_x_y_dfs(nrows, ncols)
    index = np.random.choice(nrows, nind)

    # Making reference df
    orig_ds = TensorDataset(
        torch.tensor(x[index], dtype=torch.float),
        torch.tensor(y[index], dtype=torch.float).reshape(-1, 1),
    )

    # Test 1: raw ds dl
    train_ds = MemoizedTimeSeriesDataset(
        x=x, y=y, index=index, window_frames=window_frames
    )
    check_windows(train_ds, orig_ds, window_frames)

    # Test 2: with BaseTorchModel's fit_loader
    clf = BaseTorchModel(ncols, window_frames)
    train_dl = clf.fit_loader(x, y, index, 32)
    train_ds = train_dl.dataset
    check_windows(train_ds, orig_ds, window_frames)


def test_inference_dataloaders():
    # Parameters
    nrows = 100
    ncols = 546
    window_frames = 5
    nind = 10

    # Making x, y, index
    x, y = make_x_y_dfs(nrows, ncols)
    y = np.zeros((nrows, 1))
    index = np.random.choice(nrows, nind)

    # Making reference df
    orig_ds = TensorDataset(
        torch.tensor(x[index], dtype=torch.float),
        torch.tensor(y[index], dtype=torch.float).reshape(-1, 1),
    )

    # Test 1: raw ds dl
    train_ds = TimeSeriesDataset(x=x, index=index, window_frames=window_frames)
    check_windows(train_ds, orig_ds, window_frames)

    # Test 2: with BaseTorchModel's predict_loader
    clf = BaseTorchModel(ncols, window_frames)
    test_dl = clf.predict_loader(x, index, 32)
    train_ds = test_dl.dataset
    check_windows(train_ds, orig_ds, window_frames)
