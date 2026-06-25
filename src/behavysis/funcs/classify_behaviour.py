"""Classify Behaviours."""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from behavysis.behav_classifier.behav_classifier import BehaviourClassifier
from behavysis.constants import DUR, FALSE_POS, PRED, PROB, START, STOP, TRUE_POS
from behavysis.df_classes import BehaviourPredictedDf, BehaviourScoredDf, FeaturesDf
from behavysis.models import ExperimentConfig
from behavysis.utils.io_utils import file_exists_msg


def classify_behaviour(
    features_fp: Path,
    behaviour_fp: Path,
    config_fp: Path,
    *,
    overwrite: bool,
) -> None:
    """Given model config files and features df, classifies behaviour with ML model."""
    if not overwrite and behaviour_fp.exists():
        logger.warning(file_exists_msg(behaviour_fp))
        return
    # Getting necessary config parameters
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    fps = config.auto.formatted_vid.fps
    model_config_ls = config.user.classify_behaviour
    # Getting features data
    features_df = FeaturesDf.read(features_fp)
    # Initialising y_preds df
    # Getting predictions for each classifier model and saving
    # in a list of pd.DataFrames
    behaviour_df_ls = []
    for model_config in model_config_ls:
        proj_dir = config.get_ref(model_config.proj_dir)
        behav_name = config.get_ref(model_config.behav_name)
        behav_model = BehaviourClassifier.load(proj_dir, behav_name)
        pcutoff = _get_pcutoff(
            config.get_ref(model_config.pcutoff),
            behav_model.config.pcutoff,
        )
        min_window_secs = config.get_ref(model_config.min_empty_window_secs)
        min_window_frames = int(np.round(min_window_secs * fps))
        # Running the clf pipeline
        behav_df_i = behav_model.pipeline_inference(features_df)
        # Getting prob and pred column names
        prob_col = (behav_name, PROB)
        pred_col = (behav_name, PRED)
        # Using pcutoff to get binary predictions
        behav_df_i[pred_col] = (behav_df_i[prob_col] > pcutoff).astype(int)
        # Filling in small non-behav bouts
        behav_df_i[pred_col] = _merge_bouts(behav_df_i[pred_col], min_window_frames)
        # Adding model predictions df to list
        behaviour_df_ls.append(behav_df_i)
        # Logging outcome
        logger.info("Completed %s classification.", behav_name)
    # If no models were run, then return outcome
    if len(behaviour_df_ls) == 0:
        return
    # Concatenating predictions to a single dataframe
    behaviour_df = pd.concat(behaviour_df_ls, axis=1)
    # Saving behav_preds df
    BehaviourPredictedDf.write(behaviour_df, behaviour_fp)


def _get_pcutoff(pcutoff: float, model_pcutoff: float) -> float:
    """Check if the pcutoff is valid.

    Also check if the pcutoff is the special value `-1`, in which case
    `model_pcutoff` is used.
    """
    # Checking if pcutoff is -1, then using model_pcutoff
    if pcutoff == -1:
        # Checking if model_pcutoff is valid
        assert 0 <= model_pcutoff <= 1, (
            "pcutoff is relying on the model's pcutoff.\n"
            f"But the model's pcutoff is invalid: {model_pcutoff}.\n"
            "Must be between 0 and 1."
        )
        return model_pcutoff
    assert 0 <= pcutoff <= 1, (
        "pcutoff in config must be between 0 and 1, or the special value -1.\n"
        f"Instead it has value: {pcutoff}"
    )
    return pcutoff


def _merge_bouts(vect: pd.Series, min_window_frames: int) -> pd.Series:
    """Mergs behaviour bouts that are close together.

    For a given pd.Series, `vect`,
    if the time between two bouts is less than `min_window_frames`, then merging
    the two bouts together by filling in the short `non-behav` period with `is-behav`.
    """
    vect = vect.copy()
    # Getting start, stop, and duration of each non-behav bout
    nonbouts_df = BehaviourScoredDf.vect2bouts_df(vect == FALSE_POS)
    # For each non-behav bout, if less than min_window_frames, then call it a behav
    for _, row in nonbouts_df.iterrows():
        if row[DUR] < min_window_frames:
            vect.loc[row[START] : row[STOP]] = TRUE_POS
    return vect
