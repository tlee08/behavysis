import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from behavysis.utils.logging_utils import init_logger_io_obj
from behavysis.utils.misc_utils import array2listofvect, listofvects2array


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

    def device_to_gpu(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        x_ls: list[np.ndarray],
        y_ls: list[np.ndarray],
        index_ls: list[np.ndarray],
        batch_size: int,
        epochs: int,
        val_split: float,
    ):
        logger, io_obj = init_logger_io_obj()
        # Making a 2D array of (df_index, index, y) for train test split
        index_flat = listofvects2array(index_ls, [y[index] for y, index in zip(y_ls, index_ls)])
        # Split data into training and validation sets
        index_train_flat, index_val_flat = train_test_split(index_flat, stratify=index_flat[:, 2], test_size=val_split)
        index_train_ls = array2listofvect(index_train_flat, 1)
        index_val_ls = array2listofvect(index_val_flat, 1)
        # Making data loaders
        train_dl = self.fit_loader(x_ls, y_ls, index_train_ls, batch_size=batch_size)
        val_dl = self.fit_loader(x_ls, y_ls, index_val_ls, batch_size=batch_size)
        # Storing training history
        history = pd.DataFrame(
            index=pd.Index(np.arange(epochs), name="epoch"),
            columns=["loss", "vloss"],
        )
        # Training the model
        for epoch in range(epochs):
            # Training model for one epoch
            loss = self._train_epoch(train_dl)
            # Validate model
            vloss = self._validate(val_dl)
            # showing losses
            logger.info(f"epochs: {epoch + 1}/{epochs}")
            logger.info(f"loss: {loss:.3f}, vloss: {vloss:.3f}")
            # Storing loss
            history.loc[epoch, "loss"] = loss
            history.loc[epoch, "vloss"] = vloss
        return history

    def _train_epoch(self, dl: DataLoader) -> float:
        # Switch the model to training mode
        self.train()
        # To store the running loss
        loss = 0.0
        # Iterate over the data batches
        for x_i, y_i in tqdm(dl):
            # Moving data to the device
            x_i = x_i.to(self.device)
            y_i = y_i.to(self.device)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward pass
            p_i = self(x_i)
            # Calculate loss
            loss = self.criterion(p_i, y_i)
            # Calculate backward gradients (derivative of loss w.r.t. weights)
            loss.backward()
            # Update weights based on the gradients
            self.optimizer.step()
            # Updating overall loss
            loss += loss.item()
        # Scaling loss by number of batches
        loss /= len(dl)
        # Converting loss to float
        loss = loss.cpu().detach().numpy().item()
        return loss

    def _validate(self, dl: DataLoader) -> float:
        """
        Calculating loss across an entire dataset (i.e. dataloader)
        """
        # Running inference (also returns corresponding actual labels)
        p, y = self._inference(dl)
        # Calculating the loss
        loss = self.criterion(p, y)
        # Scaling loss by number of batches
        loss /= len(dl)
        # Converting loss to float
        loss = loss.cpu().numpy().item()
        return loss

    def predict(
        self,
        x: np.ndarray,
        batch_size: int,
        index: None | np.ndarray = None,
    ) -> np.ndarray:
        # Making data loaders
        dl = self.predict_loader(x, index, batch_size=batch_size)
        # Running inference
        p, y = self._inference(dl)
        # Converting probabilities to numpy array
        p = p.cpu().numpy()
        # Flattening to 1D array
        p = p.flatten()
        return p

    def _inference(self, dl: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Given a dataloader, which has input and label tensors,
        returns the predictions and actual labels.
        """
        # Switch the model to evaluation (inference) mode
        self.eval()
        # List to store the predictions
        p = torch.zeros((len(dl.dataset), 1), device=self.device)
        y = torch.zeros((len(dl.dataset), 1), device=self.device)
        # Keeping track of number of predictions made
        n = 0
        # No need to track gradients for inference, so wrap in no_grad()
        with torch.no_grad():
            # Iterate over the data batches
            for x_i, y_i in tqdm(dl):
                # Moving data to the device
                x_i = x_i.to(self.device)
                y_i = y_i.to(self.device)
                # Running inference to get outputs
                p_i = self(x_i)
                # Storing the probabilities
                p[n : n + p_i.shape[0]] = p_i
                y[n : n + y_i.shape[0]] = y_i
                # Updating the number of predictions made
                n += p_i.shape[0]
        return p, y

    @classmethod
    def np_loader(cls, *args, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            TensorDataset(*(torch.tensor(i, dtype=torch.float) for i in args)),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def fit_loader(
        self,
        x_ls: list[np.ndarray],
        y_ls: list[np.ndarray],
        index_ls: None | list[np.ndarray] = None,
        batch_size: int = 1,
    ) -> DataLoader:
        index_ls = index_ls if index_ls is not None else [np.arange(x.shape[0]) for x in x_ls]
        ds = MemoizedTimeSeriesDataset(x_ls=x_ls, y_ls=y_ls, index_ls=index_ls, window_frames=self.window_frames)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    def predict_loader(
        self,
        x: np.ndarray,
        index: None | np.ndarray = None,
        batch_size: int = 1,
    ) -> DataLoader:
        index_ls = [index] if index is not None else [np.arange(x.shape[0])]
        ds = TimeSeriesDataset(
            x_ls=[x], y_ls=[np.zeros(x.shape[0])], index_ls=index_ls, window_frames=self.window_frames
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=False)


class TimeSeriesDataset(Dataset):
    x_ls: list[np.ndarray]
    y_ls: list[np.ndarray]
    window_frames: int

    def __init__(self, x_ls: list[np.ndarray], y_ls: list[np.ndarray], index_ls: list[np.ndarray], window_frames: int):
        # Asserting x, and y sizes are equal
        assert np.all([x.shape[0] == y.shape[0] for x, y in zip(x_ls, y_ls)])
        # Asserting indices are a valid range (between 0 and x.shape[0])
        assert np.all([np.all(index >= 0) and np.all(index < x.shape[0]) for x, index in zip(x_ls, index_ls)])
        # Padding x dfs (for frames on either side)
        x_ls = [np.concatenate([x[np.repeat(0, window_frames)], x, x[np.repeat(-1, window_frames)]]) for x in x_ls]
        # Storing the data and labels
        self.x_ls = x_ls
        self.y_ls = y_ls
        self.window_frames = window_frames
        # Making 2D array of (df_index, i)
        self.index_flat = listofvects2array(index_ls)

    def __len__(self):
        return self.index_flat.shape[0]

    def __getitem__(self, index: int):
        """
        Middle is i, start is i - window_frames, end is i + window_frames + 1.
        THe start is i and end is i + 2 * window_frames + 1.
        `i` is the index of the label. `i` is middle of data because of padding.
        """
        # Getting the dataframe index and centre index referring to x and y
        df_index, i = self.index_flat[index]
        # Getting x and y data
        x = self.x_ls[df_index]
        y = self.y_ls[df_index]
        # Getting centre for x and y data
        centre_x = i + self.window_frames
        centre_y = i
        # Getting start and end of the window
        start = centre_x - self.window_frames
        end = centre_x + self.window_frames + 1
        # Extract the window and label, and convert to torch tensors
        # Transposing from (time, features) to (features, time)
        x_i = torch.tensor(x[start:end], dtype=torch.float).transpose(1, 0)
        y_i = torch.tensor(y[centre_y], dtype=torch.float).reshape(1)
        return x_i, y_i


class MemoizedTimeSeriesDataset(TimeSeriesDataset):
    def __init__(self, x_ls: list[np.ndarray], y_ls: list[np.ndarray], index_ls: list[np.ndarray], window_frames: int):
        super().__init__(x_ls, y_ls, index_ls, window_frames)
        # For memoization
        self.memo = {}

    def __getitem__(self, index: int):
        if index in self.memo:
            # Retrieving memoized result
            return self.memo[index]
        else:
            # Otherwise, calculate
            x_i, y_i = super().__getitem__(index)
            # Memoize the result
            self.memo[index] = x_i, y_i
            return x_i, y_i
