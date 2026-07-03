"""Temporal window conversion as a pure function."""

import numpy as np


def to_windows(
    x: np.ndarray,
    window_frames: int,
) -> np.ndarray:
    """Convert row-by-row features to sliding windows.

    Parameters
    ----------
    x : np.ndarray
        Input of shape (n_samples, n_features).
    window_frames : int
        Number of frames on each side of the centre frame.

    Returns:
    -------
    np.ndarray
        Stacked windows of shape (n_samples, 2 * window_frames + 1, n_features).
    """
    if window_frames == 0:
        return x[:, np.newaxis, :]
    pad = np.pad(x, ((window_frames, window_frames), (0, 0)), mode="edge")
    return np.stack(
        [pad[i : i + 2 * window_frames + 1] for i in range(x.shape[0])],
    )
