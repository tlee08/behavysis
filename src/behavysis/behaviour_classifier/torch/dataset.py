"""WindowDataset — sliding temporal windows on __getitem__, no memoization."""

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Sliding window extraction from per-video numpy arrays.

    Pads each video at edges with edge-replicate, then slices
    a window of ``2 * window_frames + 1`` frames on __getitem__.
    No memoization — fast enough for both training and inference.
    """

    x_ls: list[np.ndarray]
    y_ls: list[np.ndarray]
    window_frames: int

    def __init__(
        self,
        x_ls: list[np.ndarray],
        y_ls: list[np.ndarray],
        index_ls: list[np.ndarray],
        window_frames: int,
    ) -> None:
        assert len(x_ls) == len(y_ls) == len(index_ls)
        assert all(
            x.shape[0] == y.shape[0] and idx.min() >= 0 and idx.max() < x.shape[0]
            for x, y, idx in zip(x_ls, y_ls, index_ls, strict=True)
        )

        self.x_ls = [
            np.pad(x, ((window_frames, window_frames), (0, 0)), mode="edge")
            for x in x_ls
        ]
        self.y_ls = y_ls
        self.window_frames = window_frames

        self._df_index = np.concatenate(
            [np.full(len(idx), i) for i, idx in enumerate(index_ls)],
        )
        self._row_index = np.concatenate(index_ls)

    def __len__(self) -> int:
        return len(self._df_index)

    def __getitem__(self, idx: int):
        df_i = self._df_index[idx]
        row_i = self._row_index[idx]

        x = self.x_ls[df_i]
        y = self.y_ls[df_i]

        centre = row_i + self.window_frames
        start = centre - self.window_frames
        end = centre + self.window_frames + 1

        x_i = torch.tensor(x[start:end], dtype=torch.float32).T  # (features, time)
        y_i = torch.tensor(y[row_i], dtype=torch.float32).reshape(1)
        return x_i, y_i
