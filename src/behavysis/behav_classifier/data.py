"""Data loading and preprocessing for behavioral classifier."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from behavysis.constants import Folders
from behavysis.df_classes.behav_classifier_df import BehavClassifierCombinedDf
from behavysis.df_classes.behav_df import BehavScoredDf, BehavValues
from behavysis.df_classes.features_df import FeaturesDf
from behavysis.utils.df_mixin import DFMixin
from behavysis.utils.io_utils import async_read_files_run, get_name, joblib_dump, joblib_load
from behavysis.utils.misc_utils import array2listofvect, listofvects2array

logger = logging.getLogger(__name__)


def combine_dfs(src_dir: Path) -> pd.DataFrame:
    """Combine all dataframes in directory into a single multi-indexed dataframe.

    Parameters
    ----------
    src_dir : Path
        Directory containing dataframe files.

    Returns
    -------
    pd.DataFrame
        Combined dataframe with experiment names as index level.
    """
    data_dict = {get_name(i): DFMixin.read(src_dir / i) for i in src_dir.iterdir()}
    df = pd.concat(data_dict.values(), axis=0, keys=data_dict.keys())
    df = BehavClassifierCombinedDf.basic_clean(df)
    return df


def wrangle_columns_y(y: pd.DataFrame) -> pd.DataFrame:
    """Filter y dataframe to actual columns and rename to `{behav}__{outcome}` format.

    Parameters
    ----------
    y : pd.DataFrame
        Scored behaviors dataframe.

    Returns
    -------
    pd.DataFrame
        Wrangled dataframe with simplified column names.
    """
    # Filtering out the pred columns (in the `outcomes` level)
    columns_filter = np.isin(
        y.columns.get_level_values(BehavScoredDf.CN.OUTCOMES.value),
        [BehavScoredDf.OutcomesCols.PRED.value],
        invert=True,
    )
    y = y.loc[:, columns_filter]
    # Setting the column names from `(behav, outcome)` to `{behav}__{outcome}`
    y.columns = [
        f"{behav_name}"
        if outcome_name == BehavScoredDf.OutcomesCols.ACTUAL.value
        else f"{behav_name}__{outcome_name}"
        for behav_name, outcome_name in y.columns
    ]
    return y


def preproc_x_fit(x: np.ndarray, preproc_fp: Path) -> None:
    """Fit preprocessing pipeline on features.

    Pipeline steps:
    - Select derived features (skip first 48 x-y-l columns)
    - MinMax scaling

    Parameters
    ----------
    x : np.ndarray
        Feature array of shape (samples, features).
    preproc_fp : Path
        Path to save fitted pipeline.
    """
    preproc_pipe = Pipeline(
        steps=[
            ("select_columns", FunctionTransformer(_select_derived_features)),
            ("min_max_scaler", MinMaxScaler()),
        ]
    )
    preproc_pipe.fit(x)
    joblib_dump(preproc_pipe, preproc_fp)


def preproc_x_transform(x: np.ndarray, preproc_fp: Path) -> np.ndarray:
    """Apply fitted preprocessing pipeline to features.

    Parameters
    ----------
    x : np.ndarray
        Feature array to transform.
    preproc_fp : Path
        Path to fitted pipeline.

    Returns
    -------
    np.ndarray
        Transformed features.
    """
    preproc_pipe: Pipeline = joblib_load(preproc_fp)
    return preproc_pipe.transform(x)


def _select_derived_features(x: np.ndarray) -> np.ndarray:
    """Select only derived features, excluding raw x-y-l coordinates.

    First 48 columns: 2 indivs * 8 bodyparts * 3 coords (x, y, likelihood)
    """
    return x[:, 48:]


def oversample(x: np.ndarray, y: np.ndarray, ratio: float) -> np.ndarray:
    """Oversample positive class to achieve target ratio.

    Parameters
    ----------
    x : np.ndarray
        Feature array.
    y : np.ndarray
        Label array.
    ratio : float
        Target ratio of positive to negative samples.

    Returns
    -------
    np.ndarray
        Resampled feature array.
    """
    assert x.shape[0] == y.shape[0]
    index = np.arange(y.shape[0])
    t = index[y == BehavValues.BEHAV.value]
    f = index[y == BehavValues.NON_BEHAV.value]
    new_t_size = int(np.round(f.shape[0] * ratio))
    t = np.random.choice(t, size=new_t_size, replace=True)
    new_index = np.concatenate([t, f])
    return x[new_index]


def undersample(x: np.ndarray, y: np.ndarray, ratio: float) -> np.ndarray:
    """Undersample negative class to achieve target ratio.

    Parameters
    ----------
    x : np.ndarray
        Feature array.
    y : np.ndarray
        Label array.
    ratio : float
        Target ratio of positive to negative samples.

    Returns
    -------
    np.ndarray
        Resampled feature array.
    """
    assert x.shape[0] == y.shape[0]
    index = np.arange(y.shape[0])
    t = index[y == BehavValues.BEHAV.value]
    f = index[y == BehavValues.NON_BEHAV.value]
    new_f_size = int(np.round(t.shape[0] / ratio))
    f = np.random.choice(f, size=new_f_size, replace=False)
    new_index = np.concatenate([t, f])
    return x[new_index]


def prepare_training_data(
    x_dir: Path,
    y_dir: Path,
    behav_name: str,
    preproc_fp: Path,
    test_split: float,
    oversample_ratio: float,
    undersample_ratio: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Load and prepare training data from features and scored behaviors.

    Parameters
    ----------
    x_dir : Path
        Directory containing feature files.
    y_dir : Path
        Directory containing scored behavior files.
    behav_name : str
        Name of behavior to train on.
    preproc_fp : Path
        Path to save preprocessing pipeline.
    test_split : float
        Fraction of data for testing.
    oversample_ratio : float
        Ratio for oversampling positive class.
    undersample_ratio : float
        Ratio for undersampling negative class.

    Returns
    -------
    tuple
        (x_ls, y_ls, index_train_ls, index_test_ls)
    """
    # Load feature and behavior files
    x_fp_ls = [x_dir / i for i in x_dir.iterdir()]
    y_fp_ls = [y_dir / i for i in y_dir.iterdir()]
    x_df_ls = async_read_files_run(x_fp_ls, FeaturesDf.read)
    y_df_ls = async_read_files_run(y_fp_ls, BehavScoredDf.read)

    # Format y dfs: select behavior column, replace UNDETERMINED
    y_df_ls = [
        y[(behav_name, BehavScoredDf.OutcomesCols.ACTUAL.value)].replace(
            BehavValues.UNDETERMINED.value, BehavValues.NON_BEHAV.value
        )
        for y in y_df_ls
    ]

    # Align x and y indices
    index_df_ls = [
        x.index.intersection(y.index) for x, y in zip(x_df_ls, y_df_ls, strict=False)
    ]
    x_df_ls = [x.loc[index] for x, index in zip(x_df_ls, index_df_ls, strict=False)]
    y_df_ls = [y.loc[index] for y, index in zip(y_df_ls, index_df_ls, strict=False)]

    assert np.all(
        [x.shape[0] == y.shape[0] for x, y in zip(x_df_ls, y_df_ls, strict=False)]
    )

    # Convert to numpy
    x_ls = [x.values for x in x_df_ls]
    y_ls = [y.values for y in y_df_ls]
    index_ls = [np.arange(x.shape[0]) for x in x_ls]

    # Fit and apply preprocessing
    preproc_x_fit(np.concatenate(x_ls, axis=0), preproc_fp)
    x_ls = [preproc_x_transform(x, preproc_fp) for x in x_ls]

    # Train-test split
    index_flat = listofvects2array(index_ls, y_ls)
    index_train_flat, index_test_flat = train_test_split(
        index_flat,
        test_size=test_split,
        stratify=index_flat[:, 2],
    )

    # Resample training data
    index_train_flat = oversample(index_train_flat, index_train_flat[:, 2], oversample_ratio)
    index_train_flat = undersample(index_train_flat, index_train_flat[:, 2], undersample_ratio)

    # Reshape back to per-dataframe lists
    index_train_ls = array2listofvect(index_train_flat, 1)
    index_test_ls = array2listofvect(index_test_flat, 1)

    return x_ls, y_ls, index_train_ls, index_test_ls
