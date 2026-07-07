"""Data loading and splitting for behavioural classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold

from behavysis.constants import FALSE_POS, UNSURE
from behavysis.utils.io_utils import async_read_files_run


def load_features(x_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    """Load feature files as numpy arrays.

    Returns:
        (x_ls, names_ls) — per-experiment arrays and experiment names.
    """
    fp_ls = sorted(x_dir.iterdir())
    names = [fp.stem for fp in fp_ls]

    def _read(fp: Path) -> np.ndarray:
        return pl.read_parquet(fp).to_pandas().set_index("frame").to_numpy()

    x_ls = async_read_files_run(fp_ls, _read)
    return x_ls, names


def load_feature_names(x_dir: Path) -> list[str]:
    """Load feature column names from the first features parquet file.

    Returns column names excluding "frame".
    """
    fp_ls = sorted(x_dir.iterdir())
    if not fp_ls:
        return []
    df = pl.read_parquet(fp_ls[0])
    return [c for c in df.columns if c != "frame"]


def load_labels(
    y_dir: Path,
    behaviour_name: str,
) -> tuple[list[np.ndarray], list[str]]:
    """Load scored behaviour labels as per-experiment numpy arrays.

    Returns:
        (y_ls, names_ls) — aligned with features.
    """
    fp_ls = sorted(y_dir.iterdir())
    names = [fp.stem for fp in fp_ls]

    def _read(fp: Path) -> np.ndarray:
        df_pl = pl.read_parquet(fp)
        df_pd = df_pl.to_pandas()
        id_vars = ["frame", "behaviour"]
        value_vars = [c for c in df_pd.columns if c not in id_vars]
        melted = df_pd.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="outcome",
            value_name="value",
        )
        pivoted = melted.pivot_table(
            index="frame",
            columns=["behaviour", "outcome"],
            values="value",
        )
        pivoted.columns = pivoted.columns.map(
            lambda x: f"{x[0]}__{x[1]}" if x[1] != "actual" else x[0],
        )
        y = pivoted[behaviour_name].replace(UNSURE, FALSE_POS).to_numpy()
        return y.reshape(-1)

    y_ls = async_read_files_run(fp_ls, _read)
    return y_ls, names


def align_features_labels(
    x_ls: list[np.ndarray],
    y_ls: list[np.ndarray],
    x_names: list[str],
    y_names: list[str],
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Align x and y arrays by finding common experiment names.

    Returns filtered (x_ls, y_ls, names).
    """
    common = sorted(set(x_names) & set(y_names))
    x_idx = [x_names.index(n) for n in common]
    y_idx = [y_names.index(n) for n in common]
    return [x_ls[i] for i in x_idx], [y_ls[i] for i in y_idx], common


def stratified_split_by_video(
    x_ls: list[np.ndarray],
    y_ls: list[np.ndarray],
    test_size: float,
    random_state: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split per-video arrays into train/test indices, stratified by label.

    Uses StratifiedGroupKFold with each video as a group.

    Returns:
        (train_idx_per_vid, test_idx_per_vid) — each list of np.ndarray row indices.
    """
    groups = np.concatenate([np.full(len(x), i) for i, x in enumerate(x_ls)])
    X = np.concatenate(x_ls, axis=0)
    y = np.concatenate(y_ls, axis=0)

    n_splits = max(2, int(1 / test_size))
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    train_idx, test_idx = next(sgkf.split(X, y, groups))

    offsets = np.cumsum([0] + [x.shape[0] for x in x_ls[:-1]])
    train_per_vid = [
        train_idx[
            (train_idx >= offsets[i]) & (train_idx < offsets[i] + x_ls[i].shape[0])
        ]
        - offsets[i]
        for i in range(len(x_ls))
    ]
    test_per_vid = [
        test_idx[(test_idx >= offsets[i]) & (test_idx < offsets[i] + x_ls[i].shape[0])]
        - offsets[i]
        for i in range(len(x_ls))
    ]
    return train_per_vid, test_per_vid
