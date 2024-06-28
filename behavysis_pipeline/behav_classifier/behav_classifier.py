"""
_summary_
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import TYPE_CHECKING, Callable, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from behavysis_core.constants import (
    BehavCN,
    BehavColumns,
    Folders,
)
from behavysis_core.data_models.pydantic_base_model import PydanticBaseModel
from behavysis_core.mixins.behav_mixin import BehavMixin
from behavysis_core.mixins.df_io_mixin import DFIOMixin
from pydantic import ConfigDict
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from .clf_templates import CLF_TEMPLATES

if TYPE_CHECKING:
    from behavysis_pipeline.pipeline.project import Project

X_ID = "x"
Y_ID = "y"

BEHAV_MODELS_SUBDIR = "behav_models"

GENERIC_BEHAV_LABELS = ["nil", "behav"]


class BehavClassifierConfigs(PydanticBaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    train_fraction: float = 0.8
    undersampling_strategy: float = 0.1
    seed: int = 42
    pcutoff: float = 0.5
    behaviour_name: str = "BehaviourName"

    window_frames: int = 5

    clf_structure: str = "clf"  # Classifier type (defined in ClfTemplates)
    batch_size: int = 256
    epochs: int = 20


class BehavClassifier:
    """
    BehavClassifier abstract class peforms behav classifier model preparation, training, saving,
    evaluation, and inference.

    Attributes
    ----------
    configs_fp: str
        _description_
    clf: BaseEstimator
        _description_
    """

    configs_fp: str
    clf: BaseEstimator

    def __init__(self, configs_fp: str) -> None:
        """
        Make a BehavClassifier instance.

        Parameters
        ----------
        configs_fp : str
            _description_
        """
        # Storing configs json fp
        self.configs_fp = configs_fp
        self.clf = None
        # Trying to read in configs json. Making a new one if it doesn't exist
        try:
            configs = BehavClassifierConfigs.read_json(self.configs_fp)
            logging.info("Reading existing model configs")
        except FileNotFoundError:
            configs = BehavClassifierConfigs()
            logging.info("Making new model configs")
        # Saving configs
        configs.write_json(self.configs_fp)

    #################################################
    # CREATE MODEL METHODS
    #################################################

    @classmethod
    def create_from_project(cls, proj: Project) -> list[BehavClassifier]:
        """
        Loading classifier from given Project instance.

        Parameters
        ----------
        proj : Project
            The Project instance.

        Returns
        -------
        BehavClassifier
            The loaded BehavClassifier instance.
        """
        # Getting the list of behaviours
        y_df = cls.wrangle_columns_y(
            cls.combine(os.path.join(proj.root_dir, Folders.SCORED_BEHAVS.value))
        )
        # For each behaviour, making a new BehavClassifier instance
        behavs_ls = y_df.columns.to_list()
        models_dir = os.path.join(proj.root_dir, BEHAV_MODELS_SUBDIR)
        models_ls = [cls.create_new_model(models_dir, behav) for behav in behavs_ls]
        # Importing data from project to "beham_models" folder (only need one model for this)
        if len(models_ls) > 0:
            models_ls[0].import_data(
                os.path.join(proj.root_dir, Folders.FEATURES_EXTRACTED.value),
                os.path.join(proj.root_dir, Folders.SCORED_BEHAVS.value),
                False,
            )
        return models_ls

    @classmethod
    def create_new_model(cls, root_dir: str, behaviour_name: str) -> BehavClassifier:
        """
        Creating a new BehavClassifier model in the given directory
        """
        configs_fp = os.path.join(root_dir, f"{behaviour_name}.json")
        # Making new BehavClassifier instance
        inst = cls(configs_fp)
        # Updating configs with project data
        configs = inst.configs
        configs.behaviour_name = behaviour_name
        configs.write_json(inst.configs_fp)
        # Returning model
        return inst

    def create_from_model(self, root_dir: str, behaviour_name: str) -> BehavClassifier:
        """
        Creating a new BehavClassifier model in the given directory
        """
        configs_fp = os.path.join(root_dir, f"{behaviour_name}.json")
        # Making new BehavClassifier instance
        inst = self.create_new_model(configs_fp, behaviour_name)
        # Using current instance's configs (but using given behaviour_name)
        configs = self.configs
        configs.behaviour_name = behaviour_name
        configs.write_json(inst.configs_fp)
        # Returning model
        return inst

    #################################################
    #            READING MODEL
    #################################################

    @classmethod
    def load(cls, configs_fp: str) -> BehavClassifier:
        """
        Reads the model from the expected model file.
        """
        if not os.path.isfile(configs_fp):
            raise FileNotFoundError(f"The model file does not exist: {configs_fp}")
        return cls(configs_fp)

    #################################################
    #            GETTER AND SETTERS
    #################################################

    @property
    def configs(self) -> BehavClassifierConfigs:
        """Returns the config model from the expected config file."""
        return BehavClassifierConfigs.read_json(self.configs_fp)

    @property
    def root_dir(self) -> str:
        """Returns the model's root directory"""
        return os.path.dirname(self.configs_fp)

    @property
    def clf_fp(self) -> str:
        """Returns the model's filepath"""
        path = os.path.join(self.root_dir, self.configs.behaviour_name, "model.sav")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    @property
    def preproc_fp(self) -> str:
        """Returns the model's preprocessor filepath"""
        path = os.path.join(self.root_dir, self.configs.behaviour_name, "preproc.sav")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def eval_dir(self) -> str:
        """Returns the model's evaluation directory"""
        path = os.path.join(self.root_dir, self.configs.behaviour_name, "eval")
        os.makedirs(path, exist_ok=True)
        return path

    #################################################
    #            IMPORTING DATA TO MODEL
    #################################################

    def import_data(self, x_dir: str, y_dir: str, overwrite=False) -> None:
        """
        Importing data from extracted features and labelled behaviours dataframes.

        Parameters
        ----------
        x_dir : str
            _description_
        y_dir : str
            _description_
        """
        # For each x and y directory
        for in_dir, id in ((x_dir, X_ID), (y_dir, Y_ID)):
            out_dir = os.path.join(self.root_dir, id)
            os.makedirs(out_dir, exist_ok=True)
            # Copying each file to model root directory
            for fp in os.listdir(in_dir):
                in_fp = os.path.join(in_dir, fp)
                out_fp = os.path.join(out_dir, fp)
                # If not overwriting and out file already exists, then skip
                if not overwrite and os.path.exists(out_fp):
                    continue
                # Copying file
                shutil.copyfile(in_fp, out_fp)

    #################################################
    #            COMBINING DFS TO SINGLE DF
    #################################################

    @staticmethod
    def combine(src_dir):
        data_dict = {
            os.path.splitext(i)[0]: pd.read_feather(os.path.join(src_dir, i))
            for i in os.listdir(os.path.join(src_dir))
        }
        return pd.concat(data_dict.values(), axis=0, keys=data_dict.keys())

    def combine_dfs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Combines the data into a single `X` df, `y` df, and index.
        The indexes of `x` and `y` will be the same (with an inner join)

        Returns
        -------
        x : pd.DataFrame
            Features dataframe of all experiments in the `x` directory
        y : pd.DataFrame
            Outcomes dataframe of all experiments in the `y` directory
        """
        # Getting the x and y dfs
        x = BehavClassifier.combine(os.path.join(self.root_dir, X_ID))
        y = BehavClassifier.combine(os.path.join(self.root_dir, Y_ID))
        # Getting the intersection pf the x and y row indexes
        index = x.index.intersection(y.index)
        x = x.loc[index]
        y = y.loc[index]
        # Returning the x and y dfs
        return x, y

    #################################################
    #            PREPROCESSING DFS
    #################################################

    @staticmethod
    def preproc_x_fit(x: np.ndarray, preproc_fp: str) -> None:
        """
        __summary__
        """
        # Making pipeline
        preproc_pipe = Pipeline(steps=[("MinMaxScaler", MinMaxScaler())])
        # Fitting pipeline
        preproc_pipe.fit(x)
        # Saving pipeline
        joblib.dump(preproc_pipe, preproc_fp)

    @staticmethod
    def preproc_x(x: np.ndarray, preproc_fp: str) -> np.ndarray:
        """
        The preprocessing steps are:
        - MinMax scaling (using previously fitted MinMaxScaler)
        """
        # Loading in pipeline
        preproc_pipe = joblib.load(preproc_fp)
        # Uses trained fit for preprocessing new data
        x = preproc_pipe.transform(x)
        # Returning df
        return x

    @staticmethod
    def wrangle_columns_y(y: pd.DataFrame) -> pd.DataFrame:
        # Filtering out the prob and pred columns (in the `outcomes` level)
        cols_filter = np.isin(
            y.columns.get_level_values(BehavCN.OUTCOMES.value),
            [BehavColumns.PROB.value, BehavColumns.PRED.value],
            invert=True,
        )
        y = y.loc[:, cols_filter]
        # Converting MultiIndex columns to single columns by
        # setting the column names from `(behav, outcome)` to `{behav}__{outcome}`
        y.columns = [
            f"{i[0]}" if i[1] == BehavColumns.ACTUAL.value else f"{i[0]}__{i[1]}"
            for i in y.columns
        ]
        return y

    @staticmethod
    def preproc_y(y: np.ndarray) -> np.ndarray:
        """
        The preprocessing steps are:
        - Imputing NaN values with 0
        - Setting -1 to 0
        - Converting the MultiIndex columns from `(behav, outcome)` to `{behav}__{outcome}`,
        by expanding the `actual` and all specific outcome columns of each behav.
        """
        # Imputing NaN values with 0
        y = np.nan_to_num(y, nan=0)
        # Setting -1 to 0 (i.e. "undecided" to "no behaviour")
        y = np.maximum(y, 0)
        # Returning arr
        return y

    @staticmethod
    def undersample(y: np.ndarray, ratio: float) -> np.ndarray:
        y_index = np.arange(y.shape[0])
        index_t = y_index[y == 1]
        index_f = np.random.choice(
            y_index[y == 0], size=int(index_t.shape[0] / ratio), replace=False
        )
        index = np.union1d(index_t, index_f)
        return index

    #################################################
    #            PIPELINE FOR DATA PREP
    #################################################

    def prepare_data_training(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares the data (`x` and `y`) in the model for training.
        Data is taken from the model's `x` and `y` dirs.

        Performs the following:
        - Combining dfs from x and y directories (individual experiment data)
        - Ensures the x and y dfs have the same index, and are in the same row order
        - Preprocesses x df. Refer to `preprocess_x` for details.
        - Preprocesses y df. Refer to `preprocess_y` for details.
        - Selects the y class (given in the configs file) from the y df.
        - Resamples the index, using the y vector's values and the
            undersampling strategy and seed in the configs.
        - Makes the X windowed array, using the resampled index.
        - Makes the y outcomes array, using the resampled index.

        Returns
        -------
        x : np.ndarray
            Features array in the format: `(samples, window, features)`
        y : np.ndarray
            Outcomes array in the format: `(samples, class)`
        """
        # Combining dfs from x and y directories (individual experiment data)
        x, y = self.combine_dfs()
        # Fitting the preprocessor pipeline
        self.preproc_x_fit(x, self.preproc_fp)
        # Preprocessing x df
        x = self.preproc_x(x, self.preproc_fp)
        # Preprocessing y df
        y = self.wrangle_columns_y(y)[self.configs.behaviour_name].values
        y = self.preproc_y(y)
        # Resampling indexes using y classes
        index = self.undersample(y, self.configs.undersampling_strategy)
        # Returning x, y, and index to use
        return x, y, index

    def prepare_data(self, x: pd.DataFrame) -> np.ndarray:
        """
        Prepares novel (`x` only) data, given the `x` pd.DataFrame.

        Performs the following:
        - Preprocesses x df. Refer to `preprocess_x` for details.
        - Makes the X windowed array, for each index.

        Returns
        -------
        x : np.ndarray
            Features array in the format: `(samples, window, features)`
        """
        # Preprocessing x df
        x = self.preproc_x(x, self.preproc_fp)
        # Returning x
        return x

    #################################################
    # PIPELINE FOR CLASSIFIER TRAINING AND INFERENCE
    #################################################

    def pipeline_build(self, clf_init_f: Callable) -> None:
        """
        Makes a classifier and saves it to the model's root directory.

        Callable is a method from `ClfTemplates`.
        """
        # Preparing data
        x, y, index = self.prepare_data_training()
        # Splitting into train and test sets
        ind_train, ind_test = train_test_split(
            index,
            test_size=1 - self.configs.train_fraction,
            stratify=y[index],
        )
        # Initialising the model
        self.clf = clf_init_f()
        # Training the model
        self.clf.fit(
            x=x,
            y=y,
            index=ind_train,
            batch_size=self.configs.batch_size,
            epochs=self.configs.epochs,
        )
        # Evaluating the model
        self.clf_eval(x, y, ind_test)
        # Updating the model configs
        configs = self.configs
        configs.clf_structure = clf_init_f.__name__
        configs.write_json(self.configs_fp)
        # Saving the model to disk
        self.clf_save()

    def pipeline_run(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Given the unprocessed features dataframe, runs the model pipeline to make predictions.

        Pipeline is:
        - Preprocess `x` df. Refer to
        [behavysis_pipeline.behav_classifier.BehavClassifier.preproc_x][] for details.
        - Makes predictions and returns the predicted behaviours.
        """
        # Saving index for later
        index = x.index
        # Preprocessing features
        x = self.prepare_data(x)
        # Loading the model
        self.clf_load()
        # Making predictions
        y_eval = self.clf_predict(x, self.configs.batch_size)
        # Settings the index
        y_eval.index = index
        # Returning predictions
        return y_eval

    #################################################
    # MODEL CLASSIFIER METHODS
    #################################################

    def clf_load(self):
        """
        Loads the model stored in `<root_dir>/<behav_name>.sav` to the model attribute.
        """
        self.clf = joblib.load(self.clf_fp)

    def clf_save(self):
        """
        Saves the model's classifier to `<root_dir>/<behav_name>.sav`.
        """
        joblib.dump(self.clf, self.clf_fp)

    def clf_predict(
        self,
        x: np.ndarray,
        batch_size: int,
        index: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Making predictions using the given model and preprocessed features.
        Assumes the x array is already preprocessed.

        Parameters
        ----------
        x : np.ndarray
            Preprocessed features.

        Returns
        -------
        pd.DataFrame
            Predicted behaviour classifications. Dataframe columns are in the format:
            ```
            behaviours :  behav    behav
            outcomes   :  "prob"   "pred"
            ```
        """
        # Getting probabilities
        index = np.arange(x.shape[0]) if index is None else index
        y_probs = self.clf.predict(
            x=x,
            index=index,
            batch_size=batch_size,
        )
        # Making predictions from probabilities (and pcutoff)
        y_preds = (y_probs > self.configs.pcutoff).astype(int)
        # Making df
        pred_df = BehavMixin.init_df(index)
        pred_df[(self.configs.behaviour_name, BehavColumns.PROB.value)] = y_probs
        pred_df[(self.configs.behaviour_name, BehavColumns.PRED.value)] = y_preds
        # Returning predicted behavs
        return pred_df

    #################################################
    # COMPREHENSIVE EVALUATION FUNCTIONS
    #################################################

    def clf_eval(
        self,
        x: np.ndarray,
        y: np.ndarray,
        index: Optional[np.ndarray] = None,
        name: Optional[str] = "",
    ) -> tuple[pd.DataFrame, plt.Figure, plt.Figure, plt.Figure]:
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
        # Making eval df
        index = np.arange(x.shape[0]) if index is None else index
        y_eval = self.clf_predict(x=x, index=index, batch_size=self.configs.batch_size)
        # Getting individual columns
        y_prob = y_eval[self.configs.behaviour_name, BehavColumns.PROB.value]
        y_pred = y_eval[self.configs.behaviour_name, BehavColumns.PRED.value]
        y_true = y[index]
        # Making confusion matrix figure
        metrics_fig = self.eval_conf_matr(y_true, y_pred)
        # Making performance for different pcutoffs figure
        pcutoffs_fig = self.eval_metrics_pcutoffs(y_true, y_prob)
        # Logistic curve
        logc_fig = self.eval_logc(y_true, y_prob)
        # Saving data and figures
        DFIOMixin.write_feather(
            y_eval, os.path.join(self.eval_dir, f"{name}_eval.feather")
        )
        metrics_fig.savefig(os.path.join(self.eval_dir, f"{name}_confm.png"))
        pcutoffs_fig.savefig(os.path.join(self.eval_dir, f"{name}_pcutoffs.png"))
        logc_fig.savefig(os.path.join(self.eval_dir, f"{name}_logc.png"))
        # Return evaluations
        return y_eval, metrics_fig, pcutoffs_fig, logc_fig

    def clf_eval_compare_all(self):
        """
        Making classifier for all available templates.

        Notes
        -----
        Takes a long time to run.
        """
        # Saving existing clf
        clf = self.clf
        # Preparing data
        x, y, index = self.prepare_data_training()
        # Splitting into train and test sets
        ind_train, ind_test = train_test_split(
            index,
            test_size=1 - self.configs.train_fraction,
            stratify=y[index],
        )
        # # Adding noise
        # noise = 0.05
        # x_train += np.random.normal(0, noise, x_train.shape)
        # x_test += np.random.normal(0, noise, x_test.shape)
        # Getting eval for each classifier in ClfTemplates
        for clf_init_f in CLF_TEMPLATES:
            clf_name = clf_init_f.__name__
            # Making classifier
            self.clf = clf_init_f()
            # Training
            self.clf.fit(
                x=x,
                y=y,
                index=ind_train,
                batch_size=self.configs.batch_size,
                epochs=self.configs.epochs,
            )
            # Evaluating on test data
            self.clf_eval(x, y, index=ind_test, name=f"{clf_name}_test")
            # Evaluating on all data
            self.clf_eval(x, y, name=f"{clf_name}_all")
        # Restoring clf
        self.clf = clf

    #################################################
    # EVALUATION METRICS FUNCTIONS
    #################################################

    @staticmethod
    def eval_conf_matr(y_true: pd.Series, y_pred: pd.Series) -> plt.Figure:
        """
        __summary__
        """
        print(
            classification_report(
                y_true,
                y_pred,
                target_names=GENERIC_BEHAV_LABELS,
            )
        )
        # Making confusion matrix
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(
            confusion_matrix(y_true, y_pred),
            annot=True,
            fmt="d",
            cmap="viridis",
            cbar=False,
            xticklabels=GENERIC_BEHAV_LABELS,
            yticklabels=GENERIC_BEHAV_LABELS,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        return fig

    @staticmethod
    def eval_metrics_pcutoffs(y_true: pd.Series, y_prob: pd.Series) -> plt.Figure:
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
                target_names=GENERIC_BEHAV_LABELS,
                output_dict=True,
            )
            precisions[i] = report[GENERIC_BEHAV_LABELS[1]]["precision"]
            recalls[i] = report[GENERIC_BEHAV_LABELS[1]]["recall"]
            f1[i] = report[GENERIC_BEHAV_LABELS[1]]["f1-score"]
            accuracies[i] = report["accuracy"]
        # Making figure
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.lineplot(x=pcutoffs, y=precisions, label="precision", ax=ax)
        sns.lineplot(x=pcutoffs, y=recalls, label="recall", ax=ax)
        sns.lineplot(x=pcutoffs, y=f1, label="f1", ax=ax)
        sns.lineplot(x=pcutoffs, y=accuracies, label="accuracy", ax=ax)
        return fig

    @staticmethod
    def eval_logc(y_true: pd.Series, y_prob: pd.Series) -> plt.Figure:
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
        # Returning figure
        return fig

    @staticmethod
    def eval_bouts(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
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
        y_eval_summary["actual_bout"] = y_eval_grouped.apply(
            lambda x: x["y_true"].mean()
        )
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
