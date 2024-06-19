"""
_summary_
"""

from __future__ import annotations

import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier

from .base_torch_model import BaseTorchModel


class RF1(RandomForestClassifier):
    def __init__(self, input_shape):
        """
        x features is (samples, features).
        y outcome is (samples, class).
        """
        super().__init__(
            n_estimators=2000,
            max_depth=3,
            random_state=0,
            n_jobs=16,
            verbose=1,
        )
        self.input_shape = input_shape

    def fit(self, x, y, *args, **kwargs):
        super().fit(x, y)
        return self

    def predict(self, x):
        return super().predict_proba(x)[:, 1]


class DNN1(BaseTorchModel):
    def __init__(self, input_shape):
        super().__init__()
        # Define the layers
        self.fc1 = nn.Linear(input_shape, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid1 = nn.Sigmoid()
        # Define the loss function and optimizer
        self.criterion: nn.Module = nn.BCELoss()
        self.optimizer: optim.Optimizer = optim.Adam(self.parameters())
        # Setting the device (GPU or CPU)
        self.device = self.device

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.sigmoid1(out)
        return out


class DNN2(BaseTorchModel):
    def __init__(self, input_shape):
        super().__init__()
        # Define the layers
        self.fc1 = nn.Linear(input_shape, 32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid1 = nn.Sigmoid()
        # Define the loss function and optimizer
        self.criterion: nn.Module = nn.BCELoss()
        self.optimizer: optim.Optimizer = optim.Adam(self.parameters())
        # Setting the device (GPU or CPU)
        self.device = self.device

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.sigmoid1(out)
        return out


class DNN3(BaseTorchModel):
    def __init__(self, input_shape):
        super().__init__()
        # Define the layers
        self.fc1 = nn.Linear(input_shape, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid1 = nn.Sigmoid()
        # Define the loss function and optimizer
        self.criterion: nn.Module = nn.BCELoss()
        self.optimizer: optim.Optimizer = optim.Adam(self.parameters())
        # Setting the device (GPU or CPU)
        self.device = self.device

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.sigmoid1(out)
        return out


class CNN1(BaseTorchModel):
    """
    x features is (samples, window, features).
    y outcome is (samples, class).
    """

    def __init__(self, input_shape):
        super().__init__()
        # Define the layers
        self.conv1 = nn.Conv1d(input_shape[2], 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * ((input_shape[1] - 2) // 2 - 2), 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid1 = nn.Sigmoid()
        # Define the loss function and optimizer
        self.criterion: nn.Module = nn.BCELoss()
        self.optimizer: optim.Optimizer = optim.Adam(self.parameters())
        # Setting the device (GPU or CPU)
        self.device = self.device

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid1(out)
        return out


CLF_TEMPLATES = [
    RF1,
    DNN1,
    DNN2,
    DNN3,
]
