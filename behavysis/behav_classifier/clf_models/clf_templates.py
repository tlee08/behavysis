"""
_summary_
"""

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier

from .base_torch_model import BaseTorchModel

# TODO: weighted loss functions


class RF1(RandomForestClassifier):
    """
    x features is (samples, features).
    y outcome is (samples, class).
    """

    def __init__(self):
        super().__init__(
            n_estimators=2000,
            max_depth=3,
            random_state=0,
            n_jobs=16,
            verbose=1,
        )
        self.window_frames = 0

    def fit(self, x_ls: list[np.ndarray], y_ls: list[np.ndarray], index_ls: list[np.ndarray], *args, **kwargs):
        # Filtering data
        x_ls = [x[index] for x, index in zip(x_ls, index_ls)]
        y_ls = [y[index] for y, index in zip(y_ls, index_ls)]
        # Concatenating dataframes
        x = np.concatenate(x_ls, axis=0)
        y = np.concatenate(y_ls, axis=0)
        # Fitting
        super().fit(x, y)
        return pd.DataFrame(
            index=pd.Index([], name="epoch"),
            columns=["loss", "vloss"],
        )

    def predict(self, x: np.ndarray, index: None | np.ndarray = None, *args, **kwargs):
        index = index if index is not None else np.arange(x.shape[0])
        return super().predict_proba(x[index])[:, 1]


class DNN1(BaseTorchModel):
    """
    x features is (samples, window, features).
    y outcome is (samples, class).
    """

    def __init__(self):
        # Initialising the parent class
        super().__init__(498, 0)  # 546
        # Input shape
        flat_size = self.window_frames * 2 + 1
        flat_size = flat_size * self.nfeatures
        # Define the layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid1 = nn.Sigmoid()
        # Define the loss function and optimizer
        self.criterion: nn.Module = nn.BCELoss()
        self.optimizer: optim.Optimizer = optim.Adam(self.parameters())
        # Setting the device (GPU or CPU)
        self.device_to_gpu()

    def forward(self, x):
        out = x
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.sigmoid1(out)
        return out


class DNN2(BaseTorchModel):
    """
    x features is (samples, window, features).
    y outcome is (samples, class).
    """

    def __init__(self):
        # Initialising the parent class
        super().__init__(498, 0)  # 546
        # Input shape
        flat_size = self.window_frames * 2 + 1
        flat_size = flat_size * self.nfeatures
        # Define the layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat_size, 32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid1 = nn.Sigmoid()
        # Define the loss function and optimizer
        self.criterion: nn.Module = nn.BCELoss()
        self.optimizer: optim.Optimizer = optim.Adam(self.parameters())
        # Setting the device (GPU or CPU)
        self.device_to_gpu()

    def forward(self, x):
        out = x
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.sigmoid1(out)
        return out


class DNN3(BaseTorchModel):
    """
    x features is (samples, window, features).
    y outcome is (samples, class).
    """

    def __init__(self):
        # Initialising the parent class
        super().__init__(498, 0)  # 546
        # Input shape
        flat_size = self.window_frames * 2 + 1
        flat_size = flat_size * self.nfeatures
        # Define the layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat_size, 256)
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
        self.device_to_gpu()

    def forward(self, x):
        out = x
        out = self.flatten(out)
        out = self.fc1(out)
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

    def __init__(self):
        # Initialising the parent class
        super().__init__(498, 10)  # 546
        # Define the layers
        self.conv1 = nn.Conv1d(self.nfeatures, 64, kernel_size=2)
        self.relu1 = nn.ReLU()
        # self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        flat_size = self.window_frames * 2 + 1
        flat_size = flat_size - 1
        # flat_size = (flat_size - 2) // 2
        flat_size = flat_size * 64

        self.fc1 = nn.Linear(flat_size, 64)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid1 = nn.Sigmoid()
        # Define the loss function and optimizer
        self.criterion: nn.Module = nn.BCELoss()
        self.optimizer: optim.Optimizer = optim.Adam(self.parameters())
        # Setting the device (GPU or CPU)
        self.device_to_gpu()

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.relu1(out)
        # out = self.maxpool1(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.sigmoid1(out)
        return out


class CNN2(BaseTorchModel):
    """
    x features is (samples, window, features).
    y outcome is (samples, class).
    """

    def __init__(self):
        # Initialising the parent class
        super().__init__(498, 10)  # 546
        # Define the layers
        self.conv1 = nn.Conv1d(self.nfeatures, 64, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        flat_size = self.window_frames * 2 + 1
        flat_size = (flat_size - 2) // 2
        flat_size = (flat_size - 2) // 2
        flat_size = flat_size * 32

        self.fc1 = nn.Linear(flat_size, 64)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid1 = nn.Sigmoid()
        # Define the loss function and optimizer
        self.criterion: nn.Module = nn.BCELoss()
        self.optimizer: optim.Optimizer = optim.Adam(self.parameters())
        # Setting the device (GPU or CPU)
        self.device_to_gpu()

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.sigmoid1(out)
        return out


CLF_TEMPLATES = [
    RF1,
    DNN1,
    DNN2,
    DNN3,
    CNN1,
    CNN2,
]
