from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm


class BaseTorchModel(nn.Module):
    criterion: nn.Module
    optimizer: optim.Optimizer
    nfeatures: int
    window_frames: int
    _device: torch.device

    def __init__(self, nfeatures: int, window_frames: int):
        super().__init__()
        # Storing the input dimensions
        self.nfeatures = nfeatures
        self.window_frames = window_frames
        # Initialising the criterion and optimizer attributes
        self.criterion = None
        self.optimizer = None
        # Setting the device (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        # Setting the device
        self._device = device
        # Updating the model to the device
        if "cpu" in self.device.type:
            self.cpu()
        if "cuda" in self.device.type:
            self.cuda(self.device)
        # TODO: allow different optimisers
        if self.optimizer:
            self.optimizer = optim.Adam(self.parameters())

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        index: np.ndarray,
        epochs: int,
        batch_size: int,
    ):
        # Making data loaders
        loader = self.fit_loader(x, y, index, batch_size=batch_size)
        # Training the model
        for epoch in range(epochs):
            # Train the model for one epoch
            loss = self._train_epoch(loader, self.criterion, self.optimizer)
            # Validate the model
            # vloss = self._validate(test_loader, criterion)
            # Print the loss
            print(f"epochs: {epoch+1}/{epochs}")
            print(f"loss: {loss:.3f}")
            # print(f"loss: {loss:.3f}, vloss: {vloss:.3f}")

    def _train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ):
        # Switch the model to training mode
        self.train()
        # Wrap the loader in tqdm to show a progress bar
        tqdm_loader = tqdm(loader)
        # Iterate over the data batches
        for data in tqdm_loader:
            running_loss = 0.0
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            outputs = self(inputs)
            # Calculate loss
            loss = criterion(outputs, labels)
            # Calculate backward gradients
            loss.backward()
            # Update learning weights
            optimizer.step()
            # print statistics
            running_loss += loss.item()
        return running_loss

    def _validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
    ):
        # Switch the model to evaluation
        self.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(self.device)
                vlabels = vlabels.to(self.device)
                voutputs = self(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        return avg_vloss

    def predict(
        self,
        x: np.ndarray,
        batch_size: int,
        index: Optional[np.ndarray] = None,
    ):
        # Making data loaders
        loader = self.predict_loader(x, index, batch_size=batch_size)
        # Running inference
        probs = self._inference(loader)
        # Returning the probabilities vector
        return probs

    def _inference(self, loader: DataLoader):
        # Switch the model to evaluation mode
        self.eval()
        # List to store the predictions
        probs_all = np.zeros(shape=(0, 1))
        # Wrap the loader in tqdm to show a progress bar
        tqdm_loader = tqdm(loader)
        # No need to track gradients for inference, so wrap in no_grad()
        with torch.no_grad():
            # Iterate over the data batches
            for data in tqdm_loader:
                inputs = data[0]
                inputs = inputs.to(self.device)
                outputs = self(inputs)
                probs = outputs.to(device="cpu", dtype=torch.float)
                probs_all = np.append(probs_all, probs.numpy(), axis=0)
        return probs_all

    @staticmethod
    def np_loader(*args, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            TensorDataset(*(torch.tensor(i, dtype=torch.float) for i in args)),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def fit_loader(
        self,
        x: np.ndarray,
        y: np.ndarray,
        index: Optional[np.ndarray] = None,
        batch_size: int = 1,
    ) -> DataLoader:
        ds = MemoizedTimeSeriesDataset(
            x=x, y=y, index=index, window_frames=self.window_frames
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=True)
        # return self.np_loader(x, y, batch_size=batch_size, shuffle=shuffle)

    def predict_loader(
        self,
        x: np.ndarray,
        index: Optional[np.ndarray] = None,
        batch_size: int = 1,
    ) -> DataLoader:
        ds = TimeSeriesDataset(x=x, index=index, window_frames=self.window_frames)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)
        # return self.np_loader(x, batch_size=batch_size, shuffle=shuffle)


class TimeSeriesDataset(Dataset):
    x: np.ndarray
    y: np.ndarray
    index: np.ndarray
    window_frames: int

    def __init__(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        index: Optional[np.ndarray] = None,
        window_frames: Optional[int] = 5,
    ):
        # Checking that the index
        if y is not None:
            assert x.shape[0] == y.shape[0]

        # Padding x (for frames on either side)
        x = self.pad_arr(x, window_frames)
        # Storing the data and labels
        self.x = x
        self.y = y if y is not None else np.zeros(x.shape[0])
        self.index = index if index is not None else np.arange(x.shape[0])
        self.window_frames = window_frames

    @staticmethod
    def pad_arr(x: np.ndarray, n: int) -> np.ndarray:
        """
        synthesising data by padding with bfill and ffill
        for window size
        """
        return np.concatenate([x[np.repeat(0, n)], x, x[np.repeat(-1, n)]])

    def __len__(self):
        return self.index.shape[0]

    def __getitem__(self, index: int):
        """
        NOTE:
        `i` is the index of the label.
        ALSO, `i` is middle of data because of padding.
        """
        # Get the actual index (because `index` is the index of `self.index`)
        i = self.index[index]
        # Calculate start and end of the window
        # Because of data padding, the start is i and end is i + 2 * window_frames + 1
        start = i
        end = i + 2 * self.window_frames + 1
        # Extract the window and label and convert to torch tensors
        x_i = torch.tensor(self.x[start:end], dtype=torch.float).transpose(1, 0)
        y_i = torch.tensor(self.y[i], dtype=torch.float).reshape(1)
        # Return
        return x_i, y_i


class MemoizedTimeSeriesDataset(TimeSeriesDataset):
    def __init__(self, x, y, index, window_frames):
        super().__init__(x, y, index, window_frames)
        # For memoization
        self.memo = {}

    def __getitem__(self, index: int):
        # Return memoized result
        if index in self.memo:
            return self.memo[index]
        else:
            # Otherwise, calculate
            window, label = super().__getitem__(index)
            # Memoize the result
            self.memo[index] = window, label
            # Return
            return window, label
