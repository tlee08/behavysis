"""Classifier architectures for behavioural classification."""

from torch import nn, optim

from .base import TorchModel


class DNN1(TorchModel):
    """Shallow feedforward network."""

    def __init__(self, nfeatures: int, window_frames: int) -> None:
        super().__init__(nfeatures, window_frames)
        flat_size = (window_frames * 2 + 1) * nfeatures
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid1 = nn.Sigmoid()

        self._criterion = nn.BCELoss()
        self._optimizer = optim.Adam(self.parameters())

    @property
    def criterion(self) -> nn.Module:
        return self._criterion

    @property
    def optimizer(self) -> optim.Optimizer:
        return self._optimizer

    def forward(self, x):
        return self.sigmoid1(self.fc2(self.dropout1(self.relu1(self.fc1(self.flatten(x))))))


class DNN2(TorchModel):
    """Shallow feedforward network with smaller hidden layer."""

    def __init__(self, nfeatures: int, window_frames: int) -> None:
        super().__init__(nfeatures, window_frames)
        flat_size = (window_frames * 2 + 1) * nfeatures
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat_size, 32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid1 = nn.Sigmoid()

        self._criterion = nn.BCELoss()
        self._optimizer = optim.Adam(self.parameters())

    @property
    def criterion(self) -> nn.Module:
        return self._criterion

    @property
    def optimizer(self) -> optim.Optimizer:
        return self._optimizer

    def forward(self, x):
        return self.sigmoid1(self.fc2(self.dropout1(self.relu1(self.fc1(self.flatten(x))))))


class DNN3(TorchModel):
    """Deeper feedforward network with two hidden layers."""

    def __init__(self, nfeatures: int, window_frames: int) -> None:
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
        self._optimizer = optim.Adam(self.parameters())

    @property
    def criterion(self) -> nn.Module:
        return self._criterion

    @property
    def optimizer(self) -> optim.Optimizer:
        return self._optimizer

    def forward(self, x):
        out = self.flatten(x)
        out = self.dropout1(self.relu1(self.fc1(out)))
        out = self.dropout2(self.relu2(self.fc2(out)))
        return self.sigmoid1(self.fc3(out))


class CNN1(TorchModel):
    """Shallow 1D convolutional network."""

    def __init__(self, nfeatures: int, window_frames: int) -> None:
        super().__init__(nfeatures, window_frames)
        self.conv1 = nn.Conv1d(nfeatures, 64, kernel_size=2)
        self.relu1 = nn.ReLU()
        self.flatten = nn.Flatten()

        flat_size = window_frames * 2 + 1
        flat_size = (flat_size - 1) * 64

        self.fc1 = nn.Linear(flat_size, 64)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid1 = nn.Sigmoid()

        self._criterion = nn.BCELoss()
        self._optimizer = optim.Adam(self.parameters())

    @property
    def criterion(self) -> nn.Module:
        return self._criterion

    @property
    def optimizer(self) -> optim.Optimizer:
        return self._optimizer

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.flatten(out)
        out = self.dropout1(self.relu3(self.fc1(out)))
        return self.sigmoid1(self.fc2(out))


class CNN2(TorchModel):
    """Deeper 1D convolutional network with pooling."""

    def __init__(self, nfeatures: int, window_frames: int) -> None:
        super().__init__(nfeatures, window_frames)
        self.conv1 = nn.Conv1d(nfeatures, 64, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        flat_size = window_frames * 2 + 1
        flat_size = (flat_size - 2) // 2
        flat_size = (flat_size - 2) // 2
        flat_size *= 32

        self.fc1 = nn.Linear(flat_size, 64)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid1 = nn.Sigmoid()

        self._criterion = nn.BCELoss()
        self._optimizer = optim.Adam(self.parameters())

    @property
    def criterion(self) -> nn.Module:
        return self._criterion

    @property
    def optimizer(self) -> optim.Optimizer:
        return self._optimizer

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = self.flatten(out)
        out = self.dropout1(self.relu3(self.fc1(out)))
        return self.sigmoid1(self.fc2(out))
