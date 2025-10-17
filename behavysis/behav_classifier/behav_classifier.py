"""
_summary_
"""

from __future__ import annotations

import os
from enum import Enum
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from behavysis.behav_classifier.clf_models.base_torch_model import (
    BaseTorchModel,
)
from behavysis.behav_classifier.clf_models.clf_templates import (
    CLF_TEMPLATES,
    CNN1,
)
from behavysis.constants import Folders
from behavysis.df_classes.behav_classifier_df import BehavClassifierCombinedDf, BehavClassifierEvalDf
from behavysis.df_classes.behav_df import BehavPredictedDf, BehavScoredDf, BehavValues
from behavysis.df_classes.features_df import FeaturesDf
from behavysis.models.behav_classifier_configs import (
    BehavClassifierConfigs,
)
from behavysis.utils.df_mixin import DFMixin
from behavysis.utils.io_utils import async_read_files_run, get_name, joblib_dump, joblib_load, write_json
from behavysis.utils.logging_utils import init_logger_file
from behavysis.utils.misc_utils import array2listofvect, enum2tuple, listofvects2array

if TYPE_CHECKING:
    from behavysis.pipeline.project import Project


class GenericBehavLabels(Enum):
    NIL = "nil"
    BEHAV = "behav"


class BehavClassifier:
    """
    BehavClassifier abstract class peforms behav classifier model preparation, training, saving,
    evaluation, and inference.
    """

    logger = init_logger_file()

    _proj_dir: str
    _behav_name: str
    _clf: BaseTorchModel

    def __init__(self, proj_dir: str, behav_name: str) -> None:
        # Setting attributes
        self._proj_dir = os.path.abspath(proj_dir)
        self._behav_name = behav_name
        # Assert that the behaviour is scored in the project (in the scored_behavs directory)
        # Getting the list of behaviours in project to check against
        y_df = self.wrangle_columns_y(self.combine_dfs(self.y_dir))
        assert np.isin(behav_name, y_df.columns)
        # Trying to load configs (or making new)
        try:
            configs = BehavClassifierConfigs.read_json(self.configs_fp)
            self.logger.debug("Loaded existing configs")
        except FileNotFoundError:
            configs = BehavClassifierConfigs()
            self.logger.debug("Made new model configs")
        # Setting and saving configs
        configs.proj_dir = self._proj_dir
        configs.behav_name = self._behav_name
        self.configs = configs
        # Trying to load classifier (or making new)
        try:
            self.clf = joblib_load(self.clf_fp)
            self.logger.debug("Loaded existing classifier")
        except FileNotFoundError:
            self.clf = CNN1()
            self.logger.debug("Made new classifier")

    #################################################
    #            GETTER AND SETTERS
    #################################################

    @property
    def proj_dir(self) -> str:
        return self._proj_dir

    @property
    def behav_name(self) -> str:
        return self._behav_name

    @property
    def clf(self) -> BaseTorchModel:
        return self._clf

    @clf.setter
    def clf(self, clf: BaseTorchModel | str) -> None:
        # If a str, then loading
        if isinstance(clf, str):
            clf_name = clf
            self._clf = joblib_load(os.path.join(self.clfs_dir, clf, "classifier.sav"))
            self.logger.debug(f"Loaded classifier: {clf_name}")
        # If a BaseTorchModel, then setting
        else:
            clf_name = type(clf).__name__
            self._clf = clf
            self.logger.debug(f"Initialised classifier: {clf_name}")
        # Updating in configs
        configs = self.configs
        configs.clf_struct = clf_name
        self.configs = configs

    @property
    def model_dir(self) -> str:
        return os.path.join(self.proj_dir, "behav_models", self.behav_name)

    @property
    def configs_fp(self) -> str:
        return os.path.join(self.model_dir, "configs.json")

    @property
    def configs(self) -> BehavClassifierConfigs:
        return BehavClassifierConfigs.read_json(self.configs_fp)

    @configs.setter
    def configs(self, configs: BehavClassifierConfigs) -> None:
        try:
            if self.configs == configs:
                return
        except FileNotFoundError:
            pass
        self.logger.debug("Configs have changed. Updating model configs on disk")
        configs.write_json(self.configs_fp)

    @property
    def clfs_dir(self) -> str:
        return os.path.join(self.model_dir, "classifiers")

    @property
    def clf_dir(self) -> str:
        return os.path.join(self.clfs_dir, self.configs.clf_struct)

    @property
    def clf_fp(self) -> str:
        return os.path.join(self.clf_dir, "classifier.sav")

    @property
    def preproc_fp(self) -> str:
        return os.path.join(self.clf_dir, "preproc.sav")

    @property
    def eval_dir(self) -> str:
        return os.path.join(self.clf_dir, "evaluation")

    @property
    def x_dir(self) -> str:
        """
        Returns the model's x directory.
        It gets the features_extracted directory from the parent Behavysis model directory.
        """
        return os.path.join(self.proj_dir, Folders.FEATURES_EXTRACTED.value)

    @property
    def y_dir(self) -> str:
        """
        Returns the model's y directory.
        It gets the scored_behavs directory from the parent Behavysis model directory.
        """
        return os.path.join(self.proj_dir, Folders.SCORED_BEHAVS.value)

    #################################################
    # CREATE/LOAD MODEL METHODS
    #################################################

    @classmethod
    def create_from_project_dir(cls, proj_dir: str) -> list:
        """
        Loading classifier from given Behavysis project directory.
        """
        # Getting the list of behaviours (after wrangling column names)
        y_df = cls.wrangle_columns_y(cls.combine_dfs(os.path.join(proj_dir, Folders.SCORED_BEHAVS.value)))
        behavs_ls = y_df.columns.to_list()
        # For each behaviour, making a new BehavClassifier instance
        models_ls = [cls(proj_dir, behav) for behav in behavs_ls]
        return models_ls

    @classmethod
    def create_from_project(cls, proj: Project) -> list[BehavClassifier]:
        """
        Loading classifier from given Behavysis project instance.
        Wraps the `create_from_project_dir` method.
        """
        return cls.create_from_project_dir(proj.root_dir)

    @classmethod
    def load(cls, proj_dir: str, behav_name: str) -> BehavClassifier:
        """
        Reads the model from the expected model file.
        """
        # Checking that the configs file exists and is valid
        configs_fp = os.path.join(proj_dir, "behav_models", behav_name, "configs.json")
        try:
            BehavClassifierConfigs.read_json(configs_fp)
        except (FileNotFoundError, OSError):
            raise ValueError(
                f'Model in project directory, "{proj_dir}", and behav name, "{behav_name}", not found.\n'
                "Please check file path."
            )
        return cls(proj_dir, behav_name)

    ###############################################################################################
    #            COMBINING DFS TO SINGLE DF
    ###############################################################################################

    @classmethod
    def combine_dfs(cls, src_dir):
        """
        Combines the data in the given directory into a single dataframe.
        Adds a MultiIndex level to the rows, with the values as the filenames in the directory.
        """
        data_dict = {get_name(i): DFMixin.read(os.path.join(src_dir, i)) for i in os.listdir(os.path.join(src_dir))}
        df = pd.concat(data_dict.values(), axis=0, keys=data_dict.keys())
        df = BehavClassifierCombinedDf.basic_clean(df)
        return df

    ###############################################################################################
    #            PREPROCESSING DFS
    ###############################################################################################

    @staticmethod
    def _preproc_x_fit_select_cols(x: np.ndarray) -> np.ndarray:
        """
        Selects only the derived features (not the x-y-l columns).

        Used in the preprocessing pipeline.
        """
        return x[:, 48:]

    @classmethod
    def preproc_x_fit(cls, x: np.ndarray, preproc_fp: str) -> None:
        """
        The preprocessing steps are:
        - Select only the derived features (not the x-y-l columns)
            - 2 (indivs) * 8 (bpts) * 3 (coords) = 48 (columns) before derived features
        - MinMax scaling (using previously fitted MinMaxScaler)
        """
        preproc_pipe = Pipeline(
            steps=[
                ("select_columns", FunctionTransformer(cls._preproc_x_fit_select_cols)),
                ("min_max_scaler", MinMaxScaler()),
            ]
        )
        preproc_pipe.fit(x)
        joblib_dump(preproc_pipe, preproc_fp)

    @classmethod
    def preproc_x_transform(cls, x: np.ndarray, preproc_fp: str) -> np.ndarray:
        """
        Runs the preprocessing steps fitted from `preproc_x_fit` on the given `x` data.
        """
        preproc_pipe: Pipeline = joblib_load(preproc_fp)
        x_preproc = preproc_pipe.transform(x)
        return x_preproc

    @classmethod
    def wrangle_columns_y(cls, y: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the `y` dataframe to only include the `behav` column and the specific outcome columns,
        and rename the columns to be in the format `{behav}__{outcome}`.
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

    @classmethod
    def oversample(cls, x: np.ndarray, y: np.ndarray, ratio: float) -> np.ndarray:
        assert x.shape[0] == y.shape[0]
        # Getting index
        index = np.arange(y.shape[0])
        # Getting indices where y is True
        t = index[y == BehavValues.BEHAV.value]
        # Getting indices where y is False
        f = index[y == BehavValues.NON_BEHAV.value]
        # Getting intended size (as t_len / f_len = ratio)
        new_t_size = int(np.round(f.shape[0] * ratio))
        # Oversampling the True indices
        t = np.random.choice(t, size=new_t_size, replace=True)
        # Combining the True and False indices
        new_index = np.concatenate([t, f])
        # Returning the resampled x
        return x[new_index]

    @classmethod
    def undersample(cls, x: np.ndarray, y: np.ndarray, ratio: float) -> np.ndarray:
        assert x.shape[0] == y.shape[0]
        # Getting index
        index = np.arange(y.shape[0])
        # Getting indices where y is True
        t = index[y == BehavValues.BEHAV.value]
        # Getting indices where y is False
        f = index[y == BehavValues.NON_BEHAV.value]
        # Getting intended size (as t_len / f_len = ratio)
        new_f_size = int(np.round(t.shape[0] / ratio))
        # Undersampling the False indices
        f = np.random.choice(f, size=new_f_size, replace=False)
        # Combining the True and False indices
        new_index = np.concatenate([t, f])
        # Returning the resampled x
        return x[new_index]

    #################################################
    #            PIPELINE FOR DATA PREP
    #################################################

    def preproc_training(
        self,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        Prepares the data for the training pipeline.

        Performs the following:
        - Combining dfs from x and y directories (individual experiment data).
        - Ensures the x and y dfs have the same index, and are in the same row order.
        - Preprocesses x df. Refer to `preprocess_x` for details.
        - Selects the y class (given in the configs file) from the y df.
        - Preprocesses y df. Refer to `preprocess_y` for details.
        - Splits into training and test indexes.
            - The training indexes are undersampled to the ratio given in the configs.

        Returns
        -------
        A tuple containing four numpy arrays:
        - x_ls: list of each dataframe's input data.
        - y_ls: list of each dataframe's target labels.
        - index_train_ls: list of each dataframe's indexes for the training data.
        - index_test_ls: list of each dataframe's indexes for the testing data.
        """
        # Getting the lists of x and y dfs
        x_fp_ls = [os.path.join(self.x_dir, i) for i in os.listdir(os.path.join(self.x_dir))]
        y_fp_ls = [os.path.join(self.y_dir, i) for i in os.listdir(os.path.join(self.y_dir))]
        x_df_ls = async_read_files_run(x_fp_ls, FeaturesDf.read)
        y_df_ls = async_read_files_run(y_fp_ls, BehavScoredDf.read)
        # Formatting y dfs (selecting column and replacing UNDETERMINED with NON_BEHAV values)
        y_df_ls = [
            y[(self.configs.behav_name, BehavScoredDf.OutcomesCols.ACTUAL.value)].replace(
                BehavValues.UNDETERMINED.value, BehavValues.NON_BEHAV.value
            )
            for y in y_df_ls
        ]
        # Ensuring x and y dfs have the same index and are in the same row order
        index_df_ls = [x.index.intersection(y.index) for x, y in zip(x_df_ls, y_df_ls)]
        x_df_ls = [x.loc[index] for x, index in zip(x_df_ls, index_df_ls)]
        y_df_ls = [y.loc[index] for y, index in zip(y_df_ls, index_df_ls)]
        assert np.all([x.shape[0] == y.shape[0] for x, y in zip(x_df_ls, y_df_ls)])
        # Converting to numpy arrays
        x_ls = [x.values for x in x_df_ls]
        y_ls = [y.values for y in y_df_ls]
        index_ls = [np.arange(x.shape[0]) for x in x_ls]
        # x preprocessing: fitting (across all x dfs) and transforming (for each x df)
        self.preproc_x_fit(np.concatenate(x_ls, axis=0), self.preproc_fp)
        x_ls = [self.preproc_x_transform(x, self.preproc_fp) for x in x_ls]
        # Making a 2D array of (df_index, index, y) for train-test splitting, stratifying and sampling
        index_flat = listofvects2array(index_ls, y_ls)
        # Splitting into train and test indexes
        index_train_flat, index_test_flat = train_test_split(
            index_flat,
            test_size=self.configs.test_split,
            stratify=index_flat[:, 2],
        )
        # Oversampling and undersampling ONLY on training data
        index_train_flat = self.oversample(index_train_flat, index_train_flat[:, 2], self.configs.oversample_ratio)
        index_train_flat = self.undersample(index_train_flat, index_train_flat[:, 2], self.configs.undersample_ratio)
        # Reshaping back to individual df index lists
        index_train_ls = array2listofvect(index_train_flat, 1)
        index_test_ls = array2listofvect(index_test_flat, 1)
        return x_ls, y_ls, index_train_ls, index_test_ls

    #################################################
    # PIPELINE FOR CLASSIFIER TRAINING AND INFERENCE
    #################################################

    def pipeline_training(self) -> None:
        """
        Makes a classifier and saves it to the model's root directory.

        Callable is a method from `ClfTemplates`.
        """
        self.logger.info(f"Training {self.configs.clf_struct}")
        # Preparing data
        x_ls, y_ls, index_train_ls, index_test_ls = self.preproc_training()
        # Training the model
        history = self.clf.fit(
            x_ls=x_ls,
            y_ls=y_ls,
            index_ls=index_train_ls,
            batch_size=self.configs.batch_size,
            epochs=self.configs.epochs,
            val_split=self.configs.val_split,
        )
        # Saving history
        self.clf_eval_save_history(history)
        # Evaluating on train and test data
        self.clf_eval_save_performance(x_ls, y_ls, index_train_ls, "train")
        self.clf_eval_save_performance(x_ls, y_ls, index_test_ls, "test")
        # Saving model
        joblib_dump(self.clf, self.clf_fp)

    def pipeline_training_all(self):
        """
        Making classifier for all available templates.
        """
        # Saving existing clf
        clf = self.clf
        for clf_cls in CLF_TEMPLATES:
            # Initialising the model
            self.clf = clf_cls()
            # Building pipeline, which runs and saves evaluation
            self.pipeline_training()
        # Restoring clf
        self.clf = clf

    def pipeline_inference(self, x_df: pd.DataFrame) -> pd.DataFrame:
        """
        Given the unprocessed features dataframe, runs the model pipeline to make predictions.

        Pipeline is:
        - Preprocess `x` df. Refer to
        [behavysis.behav_classifier.BehavClassifier.preproc_x][] for details.
        - Makes predictions and returns the predicted behaviours.
        """
        index = x_df.index
        # Preprocessing features
        x = self.preproc_x_transform(x_df.values, self.preproc_fp)
        # Loading the model
        self.clf = joblib_load(self.clf_fp)
        # Getting probabilities
        y_prob = self.clf.predict(
            x=x,
            index=np.arange(x.shape[0]),
            batch_size=self.configs.batch_size,
        )
        # Making predictions from probabilities (and pcutoff)
        y_pred = (y_prob > self.configs.pcutoff).astype(int)
        # Making df
        pred_df = BehavPredictedDf.init_df(pd.Series(index))
        pred_df[(self.configs.behav_name, BehavPredictedDf.OutcomesCols.PROB.value)] = y_prob
        pred_df[(self.configs.behav_name, BehavPredictedDf.OutcomesCols.PRED.value)] = y_pred
        return pred_df

    #################################################
    # COMPREHENSIVE EVALUATION FUNCTIONS
    #################################################

    def clf_eval_save_history(self, history: pd.DataFrame):
        # Saving history df
        DFMixin.write(history, os.path.join(self.eval_dir, f"history.{DFMixin.IO}"))
        # Making and saving history figure
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.lineplot(data=history, ax=ax)
        fig.savefig(os.path.join(self.eval_dir, "history.png"))

    def clf_eval_save_performance(
        self,
        x_ls: list[np.ndarray],
        y_ls: list[np.ndarray],
        index_ls: list[np.ndarray],
        name: str,
    ) -> tuple[pd.DataFrame, dict, Figure, Figure, Figure]:
        """
        Evaluates the classifier performance on the given x and y data.
        Saves the `metrics_fig` and `pcutoffs_fig` to the model's root directory.

        Returns
        -------
        y_eval : pd.DataFrame
            Predicted behaviour classifications against the true labels.
        metrics_fig : mpl.Figure
            Figure showing the confusion matrix.
        pcutoffs_fig : mpl.Figure
            Figure showing the precision, recall, f1, and accuracy for different pcutoffs.
        logc_fig : mpl.Figure
            Figure showing the logistic curve for different predicted probabilities.
        """
        # Getting predictions
        y_true_ls = [y[index] for y, index in zip(y_ls, index_ls)]
        y_prob_ls = [
            self.clf.predict(x=x, index=index, batch_size=self.configs.batch_size) for x, index in zip(x_ls, index_ls)
        ]
        # Making eval vects
        y_true = np.concatenate(y_true_ls)
        y_prob = np.concatenate(y_prob_ls)
        y_pred = (y_prob > self.configs.pcutoff).astype(int)
        # Making eval_df
        eval_df = BehavPredictedDf.init_df(pd.Series(np.arange(np.concatenate(index_ls).shape[0])))
        eval_df[(self.configs.behav_name, BehavPredictedDf.OutcomesCols.PROB.value)] = y_prob
        eval_df[(self.configs.behav_name, BehavPredictedDf.OutcomesCols.PRED.value)] = y_pred
        eval_df[(self.configs.behav_name, BehavScoredDf.OutcomesCols.ACTUAL.value)] = y_true
        # Making classification report
        report_dict = self.eval_report(y_true, y_pred)
        # Making confusion matrix figure
        metrics_fig = self.eval_conf_matr(y_true, y_pred)
        # Making performance for different pcutoffs figure
        pcutoffs_fig = self.eval_metrics_pcutoffs(y_true, y_prob)
        # Logistic curve
        logc_fig = self.eval_logc(y_true, y_prob)
        # Saving data and figures
        BehavClassifierEvalDf.write(eval_df, os.path.join(self.eval_dir, f"{name}_eval.{BehavClassifierEvalDf.IO}"))
        write_json(os.path.join(self.eval_dir, f"{name}_report.json"), report_dict)
        metrics_fig.savefig(os.path.join(self.eval_dir, f"{name}_confm.png"))
        pcutoffs_fig.savefig(os.path.join(self.eval_dir, f"{name}_pcutoffs.png"))
        logc_fig.savefig(os.path.join(self.eval_dir, f"{name}_logc.png"))
        return eval_df, report_dict, metrics_fig, pcutoffs_fig, logc_fig

    #################################################
    # EVALUATION METRICS FUNCTIONS
    #################################################

    @classmethod
    def eval_report(cls, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        __summary__
        """
        return classification_report(
            y_true=y_true,
            y_pred=y_pred,
            target_names=enum2tuple(GenericBehavLabels),
            output_dict=True,
        )  # type: ignore

    @classmethod
    def eval_conf_matr(cls, y_true: np.ndarray, y_pred: np.ndarray) -> Figure:
        """
        __summary__
        """
        # Making confusion matrix
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(
            confusion_matrix(y_true, y_pred),
            annot=True,
            fmt="d",
            cmap="viridis",
            cbar=False,
            xticklabels=enum2tuple(GenericBehavLabels),
            yticklabels=enum2tuple(GenericBehavLabels),
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        return fig

    @classmethod
    def eval_metrics_pcutoffs(cls, y_true: np.ndarray, y_prob: np.ndarray) -> Figure:
        """
        __summary__
        """
        # Getting precision, recall and accuracy for different cutoffs
        pcutoffs = np.linspace(0, 1, 101)
        # Measures
        precisions = np.zeros(pcutoffs.shape[0])
        recalls = np.zeros(pcutoffs.shape[0])
        f1 = np.zeros(pcutoffs.shape[0])
        accuracies = np.zeros(pcutoffs.shape[0])
        for i, pcutoff in enumerate(pcutoffs):
            y_pred = y_prob > pcutoff
            report = classification_report(
                y_true,
                y_pred,
                target_names=enum2tuple(GenericBehavLabels),
                output_dict=True,
            )
            precisions[i] = report[GenericBehavLabels.BEHAV.value]["precision"]  # type: ignore
            recalls[i] = report[GenericBehavLabels.BEHAV.value]["recall"]  # type: ignore
            f1[i] = report[GenericBehavLabels.BEHAV.value]["f1-score"]  # type: ignore
            accuracies[i] = report["accuracy"]  # type: ignore
        # Making figure
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.lineplot(x=pcutoffs, y=precisions, label="precision", ax=ax)
        sns.lineplot(x=pcutoffs, y=recalls, label="recall", ax=ax)
        sns.lineplot(x=pcutoffs, y=f1, label="f1", ax=ax)
        sns.lineplot(x=pcutoffs, y=accuracies, label="accuracy", ax=ax)
        return fig

    @classmethod
    def eval_logc(cls, y_true: np.ndarray, y_prob: np.ndarray) -> Figure:
        """
        __summary__
        """
        y_eval = pd.DataFrame(
            {
                "y_true": y_true,
                "y_prob": y_prob,
                "y_pred": y_prob > 0.4,
                "y_true_jitter": y_true + (0.2 * (np.random.rand(len(y_prob)) - 0.5)),
            }
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(
            data=y_eval,
            x="y_prob",
            y="y_true_jitter",
            marker=".",
            s=10,
            linewidth=0,
            alpha=0.2,
            ax=ax,
        )
        # Making line of ratio of y_true outcomes for each y_prob
        pcutoffs = np.linspace(0, 1, 101)
        ratios = np.vectorize(lambda i: np.mean(i > y_eval["y_prob"]))(pcutoffs)
        sns.lineplot(x=pcutoffs, y=ratios, ax=ax)
        return fig

    @classmethod
    def eval_bouts(cls, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """
        __summary__
        """
        y_eval = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        y_eval["ids"] = np.cumsum(y_eval["y_true"] != y_eval["y_true"].shift())
        # Getting the proportion of correct predictions for each bout
        y_eval_grouped = y_eval.groupby("ids")
        y_eval_summary = pd.DataFrame(
            y_eval_grouped.apply(lambda x: (x["y_pred"] == x["y_true"]).mean()),
            columns=["proportion"],
        )
        y_eval_summary["actual_bout"] = y_eval_grouped.apply(lambda x: x["y_true"].mean())
        y_eval_summary["bout_len"] = y_eval_grouped.apply(lambda x: x.shape[0])
        y_eval_summary = y_eval_summary.sort_values("proportion")
        # # Making figure
        # fig, ax = plt.subplots(figsize=(10, 7))
        # sns.scatterplot(
        #     data=y_eval_summary,
        #     x="proportion",
        #     y="bout_len",
        #     hue="actual_bout",
        #     alpha=0.4,
        #     marker=".",
        #     s=50,
        #     linewidth=0,
        #     ax=ax,
        # )
        return y_eval_summary
