"""
Classify Behaviours
"""

import numpy as np
import pandas as pd
from behavysis_core.df_classes.behav_df import BehavColumns, BehavDf
from behavysis_core.df_classes.bouts_df import BoutsDf
from behavysis_core.df_classes.df_mixin import DFMixin
from behavysis_core.mixins.io_mixin import IOMixin
from behavysis_core.mixins.misc_mixin import MiscMixin
from behavysis_core.pydantic_models.experiment_configs import ExperimentConfigs

from behavysis_pipeline.behav_classifier.behav_classifier import BehavClassifier

# TODO: handle reading the model file whilst in multiprocessing


class ClassifyBehavs:
    """__summary__"""

    @classmethod
    @IOMixin.overwrite_check()
    def classify_behavs(
        cls,
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
        features_df = DFMixin.read_feather(features_fp)
        # Initialising y_preds df
        # Getting predictions for each classifier model and saving
        # in a list of pd.DataFrames
        df_ls = np.zeros(len(models_ls), dtype="object")
        for i, model_config in enumerate(models_ls):
            # Getting model (clf, pcutoff, min_window_frames)
            model_fp = configs.get_ref(model_config.model_fp)
            try:
                behav_model = BehavClassifier.load(model_fp)
            except FileNotFoundError:
                outcome += "WARNING: Model file not found. Skipping model.\n"
                continue
            behav_name = behav_model.configs.behaviour_name
            pcutoff = cls._check_init_pcutoff(
                configs.get_ref(model_config.pcutoff), behav_model.configs.pcutoff
            )
            min_window_frames = configs.get_ref(model_config.min_window_frames)
            # Running the clf pipeline
            df_i = behav_model.pipeline_run(features_df)
            # Getting prob and pred column names
            prob_col = (behav_name, BehavColumns.PROB.value)
            pred_col = (behav_name, BehavColumns.PRED.value)
            actual_col = (behav_name, BehavColumns.ACTUAL.value)
            # Using pcutoff to get binary predictions
            df_i[pred_col] = (df_i[prob_col] > pcutoff).astype(int)
            # Filling in small non-behav bouts
            df_i[pred_col] = cls._merge_bouts(df_i[pred_col], min_window_frames)
            # NOTE: do we need "actual" and "user-defined"?
            # Including "actual" column and setting behav frames to -1
            df_i[actual_col] = df_i[pred_col].values * np.array(-1)
            # Including user-defined sub-behav columns
            for user_behav in model_config.user_behavs:
                df_i[(behav_name, user_behav)] = 0
            # Adding model predictions df to list
            df_ls[i] = df_i
            # Logging outcome
            outcome += (
                f"Completed {behav_model.configs.behaviour_name} classification.\n"
            )
        # Concatenating predictions to a single dataframe
        behavs_df = pd.concat(df_ls, axis=1)
        # Setting the index and column names
        behavs_df.index.names = list(MiscMixin.enum2tuple(BehavDf.IN))
        behavs_df.columns.names = list(MiscMixin.enum2tuple(BehavDf.CN))
        # Checking df
        BehavDf.check_df(behavs_df)
        # Saving behav_preds df
        DFMixin.write_feather(behavs_df, out_fp)
        # Returning outcome
        return outcome

    @staticmethod
    def _check_init_pcutoff(pcutoff: float, model_pcutoff: float) -> float:
        """
        Check if the initial pcutoff is valid.

        Also check if the initial pcutoff is the special value `-1`, in which case
        `model_pcutoff` is used.
        """
        # Checking if pcutoff is -1
        if pcutoff == -1:
            # Checking if model_pcutoff is valid
            assert 0 <= model_pcutoff <= 1, (
                "pcutoff is relying on the modedel's pcutoff, "
                "but the model's pcutoff is invalid."
            )
            return model_pcutoff
        assert (
            0 <= pcutoff <= 1
        ), "init_pcutoff must be between 0 and 1, or the special value -1."
        return pcutoff

    @staticmethod
    def _merge_bouts(
        vect: pd.Series,
        min_window_frames: int,
    ) -> pd.Series:
        """
        For a given pd.Series, `vect`,
        if the time between two bouts is less than `min_window_frames`, then merging
        the two bouts together by filling in the short `non-behav` period with `is-behav`.

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
        # TODO: check this func
        vect = vect.copy()
        # Getting start, stop, and duration of each non-behav bout
        nonbouts_df = BoutsDf.vect2bouts(vect == 0)
        # For each non-behav bout, if less than min_window_frames, then call it a behav
        for _, row in nonbouts_df.iterrows():
            if row["dur"] < min_window_frames:
                vect.loc[row["start"] : row["stop"]] = 1  # type: ignore
        # Returning df
        return vect
