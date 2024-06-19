import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class BaseTorchModel(nn.Module):
    criterion: nn.Module
    optimizer: optim.Optimizer
    _device: torch.device

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int = 64,
    ):
        # Making data loaders
        loader = BaseTorchModel.np_2_loader(x, y, batch_size=batch_size, shuffle=True)
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
        # Store the running loss
        running_loss = 0.0
        for i, data in enumerate(loader):
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

    def predict(self, x: np.ndarray, batch_size: int = 64):
        # Making data loaders
        loader = BaseTorchModel.np_2_loader(x, batch_size=batch_size, shuffle=False)
        # Running inference
        probs = self._inference(loader)
        # Returning the probabilities vector
        return probs

    def _inference(self, loader: DataLoader):
        # Switch the model to evaluation mode
        self.eval()
        # List to store the predictions
        probs_all = np.zeros(shape=(0, 1))
        # No need to track gradients for inference, so wrap in no_grad()
        with torch.no_grad():
            for data in loader:
                inputs = data[0]
                inputs = inputs.to(self.device)
                outputs = self(inputs)
                probs = outputs.to(device="cpu", dtype=torch.float)
                probs_all = np.append(probs_all, probs.numpy(), axis=0)
        return probs_all

    @staticmethod
    def np_2_loader(*args, batch_size: int = 1, shuffle: bool = False):
        return DataLoader(
            TensorDataset(*(torch.tensor(i, dtype=torch.float) for i in args)),
            batch_size=batch_size,
            shuffle=shuffle,
        )
