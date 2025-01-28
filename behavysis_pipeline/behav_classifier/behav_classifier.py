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

from behavysis_pipeline.behav_classifier.clf_models.base_torch_model import (
    BaseTorchModel,
)
from behavysis_pipeline.behav_classifier.clf_models.clf_templates import (
    CLF_TEMPLATES,
    CNN1,
)
from behavysis_pipeline.constants import Folders
from behavysis_pipeline.df_classes.behav_classifier_df import BehavClassifierCombinedDf
from behavysis_pipeline.df_classes.behav_df import BehavPredictedDf, BehavScoredDf, BehavValues
from behavysis_pipeline.df_classes.df_mixin import DFMixin
from behavysis_pipeline.pydantic_models.behav_classifier import (
    BehavClassifierConfigs,
)
from behavysis_pipeline.utils.io_utils import get_name, joblib_dump, joblib_load, write_json
from behavysis_pipeline.utils.logging_utils import init_logger_file
from behavysis_pipeline.utils.misc_utils import enum2tuple

if TYPE_CHECKING:
    from behavysis_pipeline.pipeline.project import Project


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
        self._proj_dir = proj_dir
        self._behav_name = behav_name
        self._clf = None
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
        configs.proj_dir = proj_dir
        configs.behav_name = behav_name
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
            assert clf in os.listdir(self.clfs_dir), f'Classifier name,  "{clf}"" not found in "{self.clfs_dir}"'
            clf_name = clf
            self._clf = joblib_load(os.path.join(self.clfs_dir, clf, "classifier.sav"))
            self.logger.debug(f"Loaded classifier: {clf_name}")
        # If a BaseTorchModel, then setting
        else:
            clf_name = type(clf).__name__
            self._clf = clf
            self.logger.debug(f"Initialised classifier: {clf_name}")
        # Updating in configs
        self.logger.debug(f"Updating clf_struct in model configs: {clf_name}")
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
    def preproc_y_transform(cls, y: np.ndarray) -> np.ndarray:
        """
        The preprocessing steps are:
        - Imputing NaN values with 0
        - Setting -1 to 0
        - Converting the MultiIndex columns from `(behav, outcome)` to `{behav}__{outcome}`,
        by expanding the `actual` and all specific outcome columns of each behav.
        """
        # Imputing NaN values with 0
        y_preproc = np.nan_to_num(y, nan=0)
        # Setting -1 to 0 (i.e. "undecided" to "no behaviour")
        y_preproc = np.maximum(y_preproc, 0)
        return y_preproc

    @classmethod
    def undersample(cls, index: np.ndarray, y: np.ndarray, ratio: float) -> np.ndarray:
        assert index.shape[0] == y.shape[0]
        # Getting array of True indices
        t = index[y == BehavValues.BEHAV.value]
        # Getting array of False indices
        f = index[y == BehavValues.NON_BEHAV.value]
        # Undersampling the False indices
        f = np.random.choice(f, size=int(np.round(t.shape[0] / ratio)), replace=False)
        # Combining the True and False indices
        uindex = np.union1d(t, f)
        return uindex

    #################################################
    #            PIPELINE FOR DATA PREP
    #################################################

    def preproc_training(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        - x: The input data.
        - y: The target labels.
        - index_train: The indexes for the training data.
        - index_test: The indexes for the testing data.
        """
        # Getting the x and y dfs
        x = self.combine_dfs(self.x_dir)
        y = self.combine_dfs(self.y_dir)
        # Getting the intersection pf the x and y row indexes
        index = x.index.intersection(y.index)
        x = x.loc[index]
        y = y.loc[index]
        assert x.shape[0] == y.shape[0]
        # Fitting the x preprocessor pipeline and transforming the x df
        self.preproc_x_fit(x.values, self.preproc_fp)
        x_preproc = self.preproc_x_transform(x.values, self.preproc_fp)
        # Preprocessing y df
        y_preproc = self.wrangle_columns_y(y)[self.configs.behav_name]
        y_preproc = self.preproc_y_transform(y_preproc.values)
        # filtering "available" index down so frames in "outer" window for each video are excluded
        window_frames = self.clf.window_frames
        if window_frames > 0:
            index_nums = (
                index.to_frame(index=False)
                .assign(index_num=np.arange(index.shape[0]))
                .groupby(BehavClassifierCombinedDf.IN.VIDEO.value)
                .apply(lambda group: group.iloc[window_frames:-window_frames])
            )["index_num"].values
        else:
            index_nums = np.arange(index.shape[0])
        # Splitting into train and test indexes
        index_train, index_test = train_test_split(
            index_nums,
            test_size=self.configs.test_split,
            stratify=y_preproc[index_nums],
        )
        # Undersampling training index
        index_train = self.undersample(index_train, y_preproc[index_train], self.configs.undersample_ratio)
        return x_preproc, y_preproc, index_train, index_test

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
        x_preproc, y_preproc, index_train, index_test = self.preproc_training()
        # Training the model
        history = self.clf.fit(
            x=x_preproc,
            y=y_preproc,
            index=index_train,
            batch_size=self.configs.batch_size,
            epochs=self.configs.epochs,
            val_split=self.configs.val_split,
        )
        # Saving history
        self.clf_eval_save_history(history)
        # Evaluating on train and test data
        self.clf_eval_save_performance(x_preproc, y_preproc, index_train, "train")
        self.clf_eval_save_performance(x_preproc, y_preproc, index_test, "test")
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

    def pipeline_inference(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Given the unprocessed features dataframe, runs the model pipeline to make predictions.

        Pipeline is:
        - Preprocess `x` df. Refer to
        [behavysis_pipeline.behav_classifier.BehavClassifier.preproc_x][] for details.
        - Makes predictions and returns the predicted behaviours.
        """
        index = x.index
        # Preprocessing features
        x_preproc = self.preproc_x_transform(x, self.preproc_fp)
        # Loading the model
        self.clf: BaseTorchModel = joblib_load(self.clf_fp)
        # Getting probabilities
        y_prob = self.clf.predict(
            x=x_preproc,
            index=np.arange(x_preproc.shape[0]),
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
        x: np.ndarray,
        y: np.ndarray,
        index: np.ndarray,
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
        y_prob = self.clf.predict(x=x, index=index, batch_size=self.configs.batch_size)
        y_pred = (y_prob > self.configs.pcutoff).astype(int)
        y_true = y[index]
        # Making eval_df
        eval_df = BehavPredictedDf.init_df(pd.Series(index))
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
        DFMixin.write(eval_df, os.path.join(self.eval_dir, f"{name}_eval.{DFMixin.IO}"))
        write_json(os.path.join(self.eval_dir, f"{name}_report.json"), report_dict)
        metrics_fig.savefig(os.path.join(self.eval_dir, f"{name}_confm.png"))
        pcutoffs_fig.savefig(os.path.join(self.eval_dir, f"{name}_pcutoffs.png"))
        logc_fig.savefig(os.path.join(self.eval_dir, f"{name}_logc.png"))
        return eval_df, report_dict, metrics_fig, pcutoffs_fig, logc_fig

    #################################################
    # EVALUATION METRICS FUNCTIONS
    #################################################

    @classmethod
    def eval_report(cls, y_true: pd.Series, y_pred: pd.Series) -> dict:
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
    def eval_conf_matr(cls, y_true: pd.Series, y_pred: pd.Series) -> Figure:
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
    def eval_metrics_pcutoffs(cls, y_true: pd.Series, y_prob: pd.Series) -> Figure:
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
    def eval_logc(cls, y_true: pd.Series, y_prob: pd.Series) -> Figure:
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
    def eval_bouts(cls, y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
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
