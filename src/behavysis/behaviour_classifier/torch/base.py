"""Base PyTorch model with training, validation, and prediction logic."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from torch import nn, optim, utils

from .dataset import WindowDataset


class TorchModel(nn.Module, ABC):
    """Abstract PyTorch model with fit/predict.

    Subclasses define the network architecture (forward),
    criterion, and optimizer. Training/inference logic is
    handled here. nfeatures and window_frames are passed
    at construction by TorchAdapter, not hardcoded.
    """

    nfeatures: int
    window_frames: int

    def __init__(self, nfeatures: int, window_frames: int) -> None:
        """Init."""
        super().__init__()
        self.nfeatures = nfeatures
        self.window_frames = window_frames
        self._device = torch.device("cpu")

    @property
    @abstractmethod
    def criterion(self) -> nn.Module:
        """Criterion."""
        ...

    @property
    @abstractmethod
    def optimizer(self) -> optim.Optimizer:
        """Optimizer."""
        ...

    @property
    def device(self) -> torch.device:
        """Device."""
        return self._device

    def to_device(self, device: torch.device | None = None) -> None:
        """Move model to device, defaulting to GPU if available."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self.to(device)
        if len(list(self.parameters())) > 0:
            opt_class = type(self.optimizer)
            self.__dict__["optimizer"] = opt_class(self.parameters())

    def fit(
        self,
        x_ls: list[np.ndarray],
        y_ls: list[np.ndarray],
        index_ls: list[np.ndarray],
        batch_size: int,
        epochs: int,
        val_split: float,
    ) -> pd.DataFrame:
        """Train model, return per-epoch loss DataFrame."""
        self.to_device()
        train_idx, val_idx = self._train_val_split(x_ls, y_ls, index_ls, val_split)

        train_dl = self._make_loader(x_ls, y_ls, train_idx, batch_size, shuffle=True)
        val_dl = self._make_loader(x_ls, y_ls, val_idx, batch_size, shuffle=False)

        history = pd.DataFrame(
            index=pd.Index(np.arange(epochs), name="epoch"),
            columns=["loss", "vloss"],
        )

        for epoch in range(epochs):
            loss = self._train_epoch(train_dl)
            vloss = self._validate(val_dl)
            logger.info(
                f"epochs: {epoch + 1}/{epochs}, loss: {loss:.3f}, vloss: {vloss:.3f}",
            )
            history.loc[epoch, "loss"] = loss
            history.loc[epoch, "vloss"] = vloss

        return history

    def predict(
        self,
        x: np.ndarray,
        index: np.ndarray | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Return predicted probabilities for indexed samples."""
        self.to_device()
        idx = index if index is not None else np.arange(x.shape[0])
        dl = self._make_loader(
            [x], [np.zeros(x.shape[0])], [idx], batch_size, shuffle=False
        )

        self.eval()
        probs = torch.zeros((len(idx), 1), device=self.device)
        n = 0
        with torch.no_grad():
            for x_i, _ in dl:
                x_i = x_i.to(self.device)
                p_i = self(x_i)
                probs[n : n + p_i.shape[0]] = p_i
                n += p_i.shape[0]
        return probs.cpu().numpy().flatten()

    def _train_epoch(self, dl: utils.data.DataLoader) -> float:
        self.train()
        total_loss = 0.0
        for x_i, y_i in dl:
            x_i, y_i = x_i.to(self.device), y_i.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self(x_i), y_i)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dl)

    def _validate(self, dl: utils.data.DataLoader) -> float:
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_i, y_i in dl:
                x_i, y_i = x_i.to(self.device), y_i.to(self.device)
                total_loss += self.criterion(self(x_i), y_i).item()
        return total_loss / len(dl)

    def _make_loader(
        self,
        x_ls: list[np.ndarray],
        y_ls: list[np.ndarray],
        index_ls: list[np.ndarray],
        batch_size: int,
        *,
        shuffle: bool,
    ) -> utils.data.DataLoader:
        ds = WindowDataset(
            x_ls=x_ls,
            y_ls=y_ls,
            index_ls=index_ls,
            window_frames=self.window_frames,
        )
        return utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def _train_val_split(
        x_ls: list[np.ndarray],
        y_ls: list[np.ndarray],
        index_ls: list[np.ndarray],
        val_split: float,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Split per-video index lists into train/val using stratification."""
        all_x = []
        all_y = []
        for i in range(len(x_ls)):
            all_x.append(x_ls[i][index_ls[i]])
            all_y.append(y_ls[i][index_ls[i]])

        X = np.concatenate(all_x, axis=0)
        y = np.concatenate(all_y, axis=0)

        train_idx, val_idx = train_test_split(
            np.arange(X.shape[0]),
            stratify=y,
            test_size=val_split,
            random_state=42,
        )

        offsets = np.cumsum([0] + [a.shape[0] for a in all_x[:-1]])
        train_per_vid = [
            train_idx[(train_idx >= off) & (train_idx < off + len(all_x[i]))] - off
            for i, off in enumerate(offsets)
        ]
        val_per_vid = [
            val_idx[(val_idx >= off) & (val_idx < off + len(all_x[i]))] - off
            for i, off in enumerate(offsets)
        ]
        return train_per_vid, val_per_vid
