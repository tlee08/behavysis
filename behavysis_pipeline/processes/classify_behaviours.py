"""
Classify Behaviours
"""

import numpy as np
import pandas as pd
from behavysis_core.constants import BehavCN, BehavColumns, BehavIN
from behavysis_core.data_models.experiment_configs import ExperimentConfigs
from behavysis_core.mixins.behav_mixin import BehavMixin
from behavysis_core.mixins.df_io_mixin import DFIOMixin
from behavysis_core.mixins.io_mixin import IOMixin

from behavysis_pipeline.behav_classifier import BehavClassifier

# TODO: handle reading the model file whilst in multiprocessing


class ClassifyBehaviours:
    """__summary__"""

    @staticmethod
    @IOMixin.overwrite_check()
    def classify_behaviours(
        features_fp: str,
        out_fp: str,
        configs_fp: str,
        overwrite: bool,
    ) -> str:
        """
        Given model config files in the BehavClassifier format, generates beahviour predidctions
        on the given extracted features dataframe.

        Parameters
        ----------
        features_fp : str
            _description_
        out_fp : str
            _description_
        configs_fp : str
            _description_
        overwrite : bool
            Whether to overwrite the output file (if it exists).

        Returns
        -------
        str
            Description of the function's outcome.

        Notes
        -----
        The config file must contain the following parameters:
        ```
        - user
            - classify_behaviours
                - models: list[str]
        ```
        Where the `models` list is a list of `model_config.json` filepaths.
        """
        outcome = ""
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        models_ls = configs.user.classify_behaviours
        # Getting features data
        features_df = DFIOMixin.read_feather(features_fp)
        # Initialising y_preds df
        # Getting predictions for each classifier model and saving
        # in a list of pd.DataFrames
        behav_preds_ls = np.zeros(len(models_ls), dtype="object")
        for i, model_config in enumerate(models_ls):
            # Getting model (clf, pcutoff, min_window_frames)
            model_fp = configs.get_ref(model_config.model_fp)
            model = BehavClassifier.load(model_fp)
            pcutoff = configs.get_ref(model_config.pcutoff)
            pcutoff = model.configs.pcutoff if pcutoff is None else pcutoff
            min_window_frames = configs.get_ref(model_config.min_window_frames)
            # Running the clf pipeline
            df_i = model.pipeline_run(features_df)
            # Getting prob and pred column names
            prob_col = (model.configs.behaviour_name, BehavColumns.PROB.value)
            pred_col = (model.configs.behaviour_name, BehavColumns.PRED.value)
            # Using pcutoff to get binary predictions
            df_i[pred_col] = (df_i[prob_col] > pcutoff).astype(int)
            # Filling in small non-behav bouts
            df_i[pred_col] = merge_bouts(df_i[pred_col], min_window_frames)
            # Adding model predictions df to list
            behav_preds_ls[i] = df_i
            # Logging outcome
            outcome += f"Completed {model.configs.behaviour_name} classification.\n"
        # Concatenating predictions to a single dataframe
        behavs_df = pd.concat(behav_preds_ls, axis=1)
        # Setting the index and column names
        behavs_df.index.name = DFIOMixin.enum_to_list(BehavIN)
        behavs_df.columns.names = DFIOMixin.enum_to_list(BehavCN)
        # Checking df
        BehavMixin.check_df(behavs_df)
        # Saving behav_preds df
        DFIOMixin.write_feather(behavs_df, out_fp)
        # Returning outcome
        return outcome


def merge_bouts(
    vect: pd.Series,
    min_window_frames: int,
) -> pd.DataFrame:
    """
    If the time between two bouts is less than `min_window_frames`, then merging
    the two bouts together by filling in the short `non-behav` period `is-behav`.

    Parameters
    ----------
    df : pd.DataFrame
        A scored_behavs dataframe.
    min_window_frames : int
        _description_

    Returns
    -------
    pd.DataFrame
        A scored_behavs dataframe, with the merged bouts.
    """
    vect = vect.copy()
    # Getting start, stop, and duration of each non-behav bout
    nonbouts_df = BehavMixin.vect_2_bouts(vect == 0)
    # For each non-behav bout, if less than min_window_frames, then call it a behav
    for _, row in nonbouts_df.iterrows():
        if row["dur"] < min_window_frames:
            vect.loc[row["start"] : row["stop"]] = 1
    # Returning df
    return vect
