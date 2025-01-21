"""
Classify Behaviours
"""

import os

import pandas as pd

from behavysis_pipeline.behav_classifier.behav_classifier import BehavClassifier
from behavysis_pipeline.df_classes.behav_df import BehavPredictedDf, BehavScoredDf, BoutCols, OutcomesPredictedCols

# from behavysis_pipeline.df_classes.bouts_df import BoutColumns, BoutsDf
from behavysis_pipeline.df_classes.features_df import FeaturesDf
from behavysis_pipeline.pydantic_models.experiment_configs import ExperimentConfigs
from behavysis_pipeline.utils.diagnostics_utils import file_exists_msg
from behavysis_pipeline.utils.logging_utils import init_logger, logger_func_decorator
from behavysis_pipeline.utils.misc_utils import enum2tuple

# TODO: handle reading the model file whilst in multiprocessing


class ClassifyBehavs:
    """__summary__"""

    logger = init_logger(__name__)

    @classmethod
    @logger_func_decorator(logger)
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
        if not overwrite and os.path.exists(out_fp):
            return file_exists_msg(out_fp)
        outcome = ""
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        models_ls = configs.user.classify_behavs
        # Getting features data
        features_df = FeaturesDf.read(features_fp)
        # Initialising y_preds df
        # Getting predictions for each classifier model and saving
        # in a list of pd.DataFrames
        df_ls = []
        for model_config in models_ls:
            proj_dir = configs.get_ref(model_config.proj_dir)
            behav_name = configs.get_ref(model_config.behav_name)
            behav_model = BehavClassifier.load(proj_dir, behav_name)
            pcutoff = cls._get_pcutoff(configs.get_ref(model_config.pcutoff), behav_model.configs.pcutoff)
            min_window_frames = configs.get_ref(model_config.min_window_frames)
            # Running the clf pipeline
            df_i = behav_model.pipeline_run(features_df)
            # Getting prob and pred column names
            prob_col = (behav_name, OutcomesPredictedCols.PROB.value)
            pred_col = (behav_name, OutcomesPredictedCols.PRED.value)
            # Using pcutoff to get binary predictions
            df_i[pred_col] = (df_i[prob_col] > pcutoff).astype(int)
            # Filling in small non-behav bouts
            df_i[pred_col] = cls._merge_bouts(df_i[pred_col], min_window_frames)
            # Adding model predictions df to list
            df_ls.append(df_i)
            # Logging outcome
            outcome += f"Completed {behav_name} classification.\n"
        # If no models were run, then return outcome
        if len(df_ls) == 0:
            return outcome
        # Concatenating predictions to a single dataframe
        behavs_df = pd.concat(df_ls, axis=1)
        # Setting the index and column names
        behavs_df.index.names = list(enum2tuple(BehavPredictedDf.IN))
        behavs_df.columns.names = list(enum2tuple(BehavPredictedDf.CN))
        # Saving behav_preds df
        BehavPredictedDf.write_feather(behavs_df, out_fp)
        # Returning outcome
        return outcome

    @staticmethod
    def _get_pcutoff(pcutoff: float, model_pcutoff: float) -> float:
        """
        Check if the pcutoff is valid.

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
            "pcutoff in configs must be between 0 and 1, or the special value -1.\n" f"Instead it has value: {pcutoff}"
        )
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
        nonbouts_df = BehavScoredDf.vect2bouts(vect == 0)
        # For each non-behav bout, if less than min_window_frames, then call it a behav
        for _, row in nonbouts_df.iterrows():
            if row[BoutCols.DUR.value] < min_window_frames:
                vect.loc[row[BoutCols.START.value] : row[BoutCols.STOP.value]] = 1
        # Returning df
        return vect
