"""Classifier architectures for behavioural classification."""

import torch
from torch import nn, optim
from torch.nn import functional

from .base import TorchModel


class DNN1(TorchModel):
    """Shallow feedforward network."""

    def __init__(self, nfeatures: int, window_frames: int) -> None:
        """Init."""
        super().__init__(nfeatures, window_frames)
        flat_size = (window_frames * 2 + 1) * nfeatures
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid1 = nn.Sigmoid()

        self._criterion = nn.BCELoss()

    @property
    def criterion(self) -> nn.Module:
        """Criterion."""
        return self._criterion

    def _make_optimizer(self) -> optim.Optimizer:
        """Optimizer."""
        return optim.Adam(self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.sigmoid1(
            self.fc2(self.dropout1(self.relu1(self.fc1(self.flatten(x)))))
        )


class DNN2(TorchModel):
    """Shallow feedforward network with smaller hidden layer."""

    def __init__(self, nfeatures: int, window_frames: int) -> None:
        """Init."""
        super().__init__(nfeatures, window_frames)
        flat_size = (window_frames * 2 + 1) * nfeatures
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat_size, 32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid1 = nn.Sigmoid()

        self._criterion = nn.BCELoss()

    @property
    def criterion(self) -> nn.Module:
        """Criterion."""
        return self._criterion

    def _make_optimizer(self) -> optim.Optimizer:
        """Optimizer."""
        return optim.Adam(self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.sigmoid1(
            self.fc2(self.dropout1(self.relu1(self.fc1(self.flatten(x)))))
        )


class DNN3(TorchModel):
    """Deeper feedforward network with two hidden layers."""

    def __init__(self, nfeatures: int, window_frames: int) -> None:
        """Init."""
        super().__init__(nfeatures, window_frames)
        flat_size = (window_frames * 2 + 1) * nfeatures
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat_size, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid1 = nn.Sigmoid()

        self._criterion = nn.BCELoss()

    @property
    def criterion(self) -> nn.Module:
        """Criterion."""
        return self._criterion

    def _make_optimizer(self) -> optim.Optimizer:
        """Optimizer."""
        return optim.Adam(self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.dropout1(self.relu1(self.fc1(self.flatten(x))))
        out = self.dropout2(self.relu2(self.fc2(out)))
        return self.sigmoid1(self.fc3(out))


class CNN(TorchModel):
    """1D temporal CNN with global average pooling.

    Three same-padding convolutional blocks over the features-by-time input,
    collapsed to a fixed-size vector by global average pooling, then a small
    MLP head.  The pooling makes the architecture robust to any window length.
    """

    def __init__(self, nfeatures: int, window_frames: int) -> None:
        """Init."""
        super().__init__(nfeatures, window_frames)
        self.conv1 = nn.Conv1d(nfeatures, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)

        self._criterion = nn.BCELoss()

    @property
    def criterion(self) -> nn.Module:
        """Criterion."""
        return self._criterion

    def _make_optimizer(self) -> optim.Optimizer:
        """Optimizer."""
        return optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = functional.relu(self.bn1(self.conv1(x)))
        out = functional.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = functional.relu(self.bn3(self.conv3(out)))
        out = self.avgpool(out).squeeze(-1)
        out = functional.relu(self.fc1(out))
        out = self.dropout(out)
        return torch.sigmoid(self.fc2(out))


# Name → architecture class, for (de)serialisation.
MODEL_TYPES: dict[str, type[TorchModel]] = {
    "CNN": CNN,
    "DNN1": DNN1,
    "DNN2": DNN2,
    "DNN3": DNN3,
}
