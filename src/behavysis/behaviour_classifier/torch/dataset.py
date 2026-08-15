"""WindowDataset — sliding temporal windows on __getitem__."""

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Sliding temporal window over per-video feature matrices.

    Each video is edge-padded by ``window_frames`` so every frame is the
    centre of a ``2 * window_frames + 1`` window.  Yields ``(features, time)``
    tensors.
    """

    x_ls: list[np.ndarray]
    y_ls: list[np.ndarray]
    window_frames: int

    def __init__(
        self,
        x_ls: list[np.ndarray],
        y_ls: list[np.ndarray],
        window_frames: int,
    ) -> None:
        """Init."""
        if len(x_ls) != len(y_ls):
            msg = "x_ls and y_ls must have the same number of videos"
            raise ValueError(msg)
        self.x_ls = [
            np.pad(x, ((window_frames, window_frames), (0, 0)), mode="edge")
            for x in x_ls
        ]
        self.y_ls = y_ls
        self.window_frames = window_frames
        self._video_index = np.concatenate(
            [np.full(len(y), i) for i, y in enumerate(y_ls)],
        )
        self._row_index = np.concatenate([np.arange(len(y)) for y in y_ls])

    def __len__(self) -> int:
        """Number of windowed samples."""
        return len(self._video_index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(features, time)`` window and label for ``idx``."""
        v = self._video_index[idx]
        r = self._row_index[idx]
        centre = r + self.window_frames
        x_i = torch.tensor(
            self.x_ls[v][centre - self.window_frames : centre + self.window_frames + 1],
            dtype=torch.float32,
        ).T
        y_i = torch.tensor(self.y_ls[v][r], dtype=torch.float32).reshape(1)
        return x_i, y_i
