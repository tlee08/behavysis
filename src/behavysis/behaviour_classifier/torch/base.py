"""Base PyTorch sequence model with training and prediction logic."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from torch import nn, optim, utils

from .dataset import WindowDataset


class TorchModel(nn.Module, ABC):
    """Abstract PyTorch sequence model with fit/predict.

    Subclasses define the architecture (``forward``), ``criterion`` and
    ``_make_optimizer``.  Training and prediction operate on a list of
    per-experiment feature matrices (``x_ls``) and label vectors (``y_ls``),
    sliding a temporal window of ``2 * window_frames + 1`` frames over each
    experiment.  Runs on MPS/CUDA when available, else CPU.
    """

    nfeatures: int
    window_frames: int

    def __init__(self, nfeatures: int, window_frames: int) -> None:
        """Init."""
        super().__init__()
        self.nfeatures = nfeatures
        self.window_frames = window_frames
        self.optimizer: optim.Optimizer | None = None

    @property
    @abstractmethod
    def criterion(self) -> nn.Module:
        """Loss."""

    @abstractmethod
    def _make_optimizer(self) -> optim.Optimizer:
        """Fresh optimizer over this model's parameters."""

    @staticmethod
    def _default_device() -> torch.device:
        """Best available device (CUDA > MPS > CPU)."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(
        self,
        x_ls: list[np.ndarray],
        y_ls: list[np.ndarray],
        batch_size: int,
        epochs: int,
    ) -> pd.DataFrame:
        """Train for ``epochs`` epochs, returning per-epoch losses."""
        device = self._default_device()
        self.to(device)
        self.optimizer = self._make_optimizer()
        dl = self._make_loader(x_ls, y_ls, batch_size, shuffle=True)
        history = pd.DataFrame(
            index=pd.Index(np.arange(epochs), name="epoch"), columns=["loss"]
        )
        for epoch in range(epochs):
            history.loc[epoch, "loss"] = self._train_epoch(dl, device)
        return history

    def predict(self, x_ls: list[np.ndarray], batch_size: int = 256) -> np.ndarray:
        """Return per-frame probabilities, concatenated across experiments."""
        device = self._default_device()
        self.to(device)
        y_ls = [np.zeros(len(x)) for x in x_ls]
        dl = self._make_loader(x_ls, y_ls, batch_size, shuffle=False)
        self.eval()
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for x_i, _ in dl:
                probs.append(self(x_i.to(device)).cpu().numpy().reshape(-1))
        return np.concatenate(probs)

    def _train_epoch(self, dl: utils.data.DataLoader, device: torch.device) -> float:
        self.train()
        total = 0.0
        for x_i, y_i in dl:
            x = x_i.to(device)
            y = y_i.to(device)
            self.optimizer.zero_grad()
            loss = self.criterion(self(x), y)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
        return total / len(dl)

    def _make_loader(
        self,
        x_ls: list[np.ndarray],
        y_ls: list[np.ndarray],
        batch_size: int,
        *,
        shuffle: bool,
    ) -> utils.data.DataLoader:
        ds = WindowDataset(x_ls, y_ls, self.window_frames)
        return utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
