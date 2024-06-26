"""
_summary_
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import TYPE_CHECKING, Callable

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
from behavysis_core.mixins.features_mixin import FeaturesMixin
from behavysis_core.mixins.io_mixin import IOMixin
from imblearn.under_sampling import RandomUnderSampler
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
        y_df = BehavClassifier.preproc_y(
            pd.concat(
                [
                    BehavMixin.read_feather(exp.get_fp(Folders.SCORED_BEHAVS.value))
                    for exp in proj.get_experiments()
                ],
            )
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
        return os.path.split(self.configs_fp)[0]

    @property
    def clf_fp(self) -> str:
        """Returns the model's filepath"""
        return os.path.join(self.root_dir, f"{self.configs.behaviour_name}.sav")

    @property
    def preproc_fp(self) -> str:
        """Returns the model's preprocessor filepath"""
        return os.path.join(self.root_dir, f"{self.configs.behaviour_name}_preproc.sav")

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
        # data stores
        id_ls = [X_ID, Y_ID]
        df_dict = {
            X_ID: None,
            Y_ID: None,
        }
        read_func_ls = {
            X_ID: FeaturesMixin.read_feather,  # features extracted
            Y_ID: BehavMixin.read_feather,  # behaviours scored
        }
        # Reading in each x and y df and storing in data dict
        for df_id in id_ls:
            # Making a list of dfs fpr each df in the given data directory
            df_dir = os.path.join(self.root_dir, df_id)
            df_ls = np.zeros(len(os.listdir(df_dir)), dtype=object)
            for i, fp in enumerate(os.listdir(df_dir)):
                name = IOMixin.get_name(fp)
                x_fp = os.path.join(df_dir, f"{name}.feather")
                df_ls[i] = pd.concat(
                    [read_func_ls[df_id](x_fp)],
                    axis=0,
                    keys=[name],
                    names=["experiment"],
                )
            # Concatenating the list of dfs together to make the combined x and y dfs
            df_dict[df_id] = pd.concat(df_ls)
        # Getting the intersection pf the x and y row indexes
        index = df_dict[X_ID].index.intersection(df_dict[Y_ID].index)
        # Filtering on this index intersection
        x = df_dict[X_ID].loc[index]
        y = df_dict[Y_ID].loc[index]
        # Returning the x and y dfs
        return x, y

    #################################################
    #            PREPROCESSING DFS
    #################################################

    def preproc_x_fit(self, x: pd.DataFrame) -> None:
        """
        __summary__
        """
        # Making pipeline
        preproc_pipe = Pipeline(steps=[("MinMaxScaler", MinMaxScaler())])
        # Fitting pipeline
        preproc_pipe.fit(x)
        # Saving pipeline
        joblib.dump(preproc_pipe, self.preproc_fp)

    def preproc_x(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        The preprocessing steps are:
        - MinMax scaling (using previously fitted MinMaxScaler)
        """
        # Loading in pipeline
        preproc_pipe = joblib.load(self.preproc_fp)
        # Uses trained fit for preprocessing new data
        x = pd.DataFrame(
            preproc_pipe.transform(x),
            index=x.index,
            columns=x.columns,
        )
        # Returning df
        return x

    @staticmethod
    def preproc_y(y: pd.DataFrame) -> pd.DataFrame:
        """
        The preprocessing steps are:
        - Imputing NaN values with 0
        - Setting -1 to 0
        - Converting the MultiIndex columns from `(behav, outcome)` to `{behav}__{outcome}`,
        by expanding the `actual` and all specific outcome columns of each behav.
        """
        # Imputing NaN values with 0
        y = y.fillna(0)
        # Setting -1 to 0 (i.e. "undecided" to "no behaviour")
        y = y.map(lambda x: 0 if x == -1 else x)
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
        # Returning df
        return y

    def resample(self, y: pd.Series) -> pd.MultiIndex:
        """
        Uses the resampling strategy and seed in configs.
        Returns the index for resampling. This can then
        be used to filter the X and y dfs.

        Notes
        -----
        Only indices within the window frame range are considered for resampling.
        """
        index = y.index
        n = self.configs.window_frames
        # Getting only valid indexes (i.e. where window can be applied - won't go out of bounds)
        valid_index = pd.MultiIndex.from_frame(
            index.to_frame(index=False)
            .groupby("experiment")["frame"]
            .apply(lambda x: x.iloc[n:-n])
            .reset_index("experiment")
        ).sort_values()
        # Undersampling and getting resampled index
        resampled_index = pd.MultiIndex.from_frame(
            RandomUnderSampler(
                sampling_strategy=self.configs.undersampling_strategy,
                random_state=self.configs.seed,
            ).fit_resample(X=valid_index.to_frame(index=False), y=y.loc[valid_index])[0]
        ).sort_values()
        # Returning resampled index
        return resampled_index

    def make_windows(
        self, df: pd.DataFrame, resampled_index: pd.MultiIndex
    ) -> np.ndarray:
        """
        Returns np array of (samples, frames, features).

        Notes
        -----
        Anything that is on the edge of the window will be padded with bfilled and ffilled.
        This introduces synthetic data.
        """
        # NOTE: synthesising data by padding with bfill and ffill
        # for window size
        fpad = df.iloc[np.repeat(0, self.configs.window_frames)]
        fpad.index = np.repeat(None, self.configs.window_frames)
        bpad = df.iloc[np.repeat(-1, self.configs.window_frames)]
        bpad.index = np.repeat(None, self.configs.window_frames)
        df = pd.concat([fpad, df, bpad])
        # Getting array of index numbers
        resampled_arr = [df.index.get_loc(i) for i in resampled_index]
        n = self.configs.window_frames
        # Making arrays of (samples, window, features)
        return np.stack([df.iloc[i - n : i + n + 1].values for i in resampled_arr])

    def train_test_split(self, x, y):
        """
        Splitting into train and test sets
        """
        # Splitting into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=1 - self.configs.train_fraction,
            stratify=y,
        )
        # Manual split to separate bouts themselves
        # split = int(x.shape[0] * self.configs.train_fraction)
        # x_train, x_test = x[:split], x[split:]
        # y_train, y_test = y[:split], y[split:]
        return x_train, x_test, y_train, y_test

    #################################################
    #            PIPELINE FOR DATA PREP
    #################################################

    def prepare_data_training(self) -> tuple[np.ndarray, np.ndarray]:
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
        self.preproc_x_fit(x)
        # Preprocessing x df
        x = self.preproc_x(x)
        # Preprocessing y df
        y = self.preproc_y(y)
        # Selecting y class
        y = y[self.configs.behaviour_name]
        # Resampling indexes using y classes
        index = self.resample(y)
        # Making x (windowed) array
        # x = self.make_windows(x, index)
        x = x.loc[index].values
        # Making y array
        y = y.loc[index].values.reshape(-1, 1)
        # Returning x and y
        return x, y

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
        x = self.preproc_x(x)
        # Making x (windowed) array
        # x = self.make_windows(x, x.index)
        x = x.values
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
        x, y = self.prepare_data_training()
        # Splitting into train and test sets
        x_train, x_test, y_train, y_test = self.train_test_split(x, y)
        # Initialising the model
        self.clf = clf_init_f(x.shape[1])
        # Evaluating the model (training on train, testing on test)
        self.clf_train(x_train, y_train)
        self.clf_eval(x_test, y_test)
        # Training the model (on all data)
        self.clf_train(x, y)
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
        y_eval = self.clf_predict(x)
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

    def clf_train(self, x: np.array, y: np.array, **kwargs) -> None:
        """
        __summary__
        """
        self.clf.fit(
            x,
            y,
            batch_size=kwargs.get("batch_size", 128),
            epochs=kwargs.get("epochs", 200),
        )

    def clf_predict(self, x: np.ndarray, **kwargs) -> pd.DataFrame:
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
        y_probs = self.clf.predict(x)
        # Making predictions from probabilities (and pcutoff)
        y_preds = (y_probs > self.configs.pcutoff).astype(int)
        # Making df
        pred_df = BehavMixin.init_df(np.arange(x.shape[0]))
        pred_df[(self.configs.behaviour_name, BehavColumns.PROB.value)] = y_probs
        pred_df[(self.configs.behaviour_name, BehavColumns.PRED.value)] = y_preds
        # Returning predicted behavs
        return pred_df

    #################################################
    # COMPREHENSIVE EVALUATION FUNCTIONS
    #################################################

    def clf_eval(
        self, x: np.ndarray, y: np.ndarray
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
        # Making eval dir
        eval_dir = os.path.join(self.root_dir, "eval", self.configs.behaviour_name)
        name = self.configs.behaviour_name
        os.makedirs(eval_dir, exist_ok=True)
        # Making eval df
        y_eval = self.clf_predict(x)
        y_eval[(self.configs.behaviour_name, BehavColumns.ACTUAL.value)] = y
        DFIOMixin.write_feather(y_eval, os.path.join(eval_dir, f"{name}_eval.feather"))
        # Getting individual columns
        y_prob = y_eval[self.configs.behaviour_name, BehavColumns.PROB.value]
        y_pred = y_eval[self.configs.behaviour_name, BehavColumns.PRED.value]
        y_true = y_eval[self.configs.behaviour_name, BehavColumns.ACTUAL.value]
        # Making confusion matrix figure
        metrics_fig = self.eval_conf_matr(y_true, y_pred)
        metrics_fig.savefig(os.path.join(eval_dir, "confm.png"))
        # Making performance for different pcutoffs figure
        pcutoffs_fig = self.eval_metrics_pcutoffs(y_true, y_prob)
        pcutoffs_fig.savefig(os.path.join(eval_dir, "pcutoffs.png"))
        # Logistic curve
        logc_fig = self.eval_logc(y_true, y_prob)
        logc_fig.savefig(os.path.join(eval_dir, "logc.png"))
        # Return evaluations
        return y_eval, metrics_fig, pcutoffs_fig, logc_fig

    def clf_eval_compare(self, eval_name: str, x_test: np.ndarray, y_test: np.ndarray):
        """
        Making and saving evaluation metrics for the classifier, with a given name.
        """
        eval_dir = os.path.join(
            self.root_dir, "compare_evals", self.configs.behaviour_name
        )
        os.makedirs(eval_dir, exist_ok=True)
        e, m, p, l = self.clf_eval(x_test, y_test)
        e.to_feather(os.path.join(eval_dir, f"{eval_name}_e.feather"))
        m.savefig(os.path.join(eval_dir, f"{eval_name}_m.png"))
        p.savefig(os.path.join(eval_dir, f"{eval_name}_p.png"))
        l.savefig(os.path.join(eval_dir, f"{eval_name}_l.png"))

    def clf_eval_compare_all(self):
        """
        Making classifier for all available templates.

        Notes
        -----
        Takes a long time to run.
        """
        # Saving existing clf
        clf = self.clf
        # Getting data
        # train and test data
        x, y = self.prepare_data_training()
        x_train, x_test, y_train, y_test = self.train_test_split(x, y)
        # # Adding noise
        # noise = 0.05
        # x_train += np.random.normal(0, noise, x_train.shape)
        # x_test += np.random.normal(0, noise, x_test.shape)
        # All data
        x, y = self.combine_dfs()
        x = self.prepare_data(x)
        y = y[(self.configs.behaviour_name, BehavColumns.ACTUAL.value)]
        y = np.vectorize(lambda i: 0 if i == -1 else i)(y)
        # Getting eval for each classifier in ClfTemplates
        for init_f in CLF_TEMPLATES:
            # Making classifier
            self.clf = init_f(x.shape[1])
            # Training
            self.clf_train(x_train, y_train)
            # Evaluating on test data
            self.clf_eval_compare(f"{init_f.__name__}_test", x_test, y_test)
            # Evaluating on all data
            self.clf_eval_compare(f"{init_f.__name__}_all", x, y)
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
