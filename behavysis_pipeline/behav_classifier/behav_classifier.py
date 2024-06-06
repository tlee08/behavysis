"""
_summary_
"""

from __future__ import annotations

import logging
import os
import shutil
from enum import Enum
from typing import TYPE_CHECKING

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from behavysis_core.constants import (
    BEHAV_CN,
    BEHAV_IN,
    FEATURES_CN,
    FEATURES_IN,
    BehavColumns,
    Folders,
)
from behavysis_core.mixins.behav_mixin import BehavMixin
from behavysis_core.mixins.df_io_mixin import DFIOMixin
from behavysis_core.mixins.features_mixin import FeaturesMixin
from behavysis_core.mixins.io_mixin import IOMixin
from imblearn.under_sampling import RandomUnderSampler
from keras.layers import Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D
from keras.models import Model
from keras.utils import plot_model
from matplotlib.figure import Figure
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import MinMaxScaler

from behavysis_pipeline.behav_classifier.behav_classifier_configs import (
    BehavClassifierConfigs,
)

if TYPE_CHECKING:
    from behavysis_pipeline.pipeline.project import Project


class Datasets(Enum):
    ALL = "all"
    TRAIN = "train"
    TEST = "test"


X_ID = "x"
Y_ID = "y"
SUBSAMPLED = "sub"

COMB_IN = ["experiments", *BEHAV_IN]
COMB_X_CN = FEATURES_CN
CN = BEHAV_CN


class BehavClassifier:
    """
    BehavClassifier class peforms behav classifier model preparation, training, saving,
    evaluation, and inference.
    """

    configs_fp: str
    clf: Model

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
        y_df = BehavClassifier.preprocess_y(
            pd.concat(
                [
                    BehavMixin.read_feather(exp.get_fp(Folders.SCORED_BEHAVS.value))
                    for exp in proj.get_experiments()
                ],
            )
        )
        # For each behaviour, making a new BehavClassifier instance
        behavs_ls = y_df.columns.to_list()
        models_dir = os.path.join(proj.root_dir, "behav_models")
        models_ls = [cls.create_new_model(models_dir, behav) for behav in behavs_ls]
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

    #################################################
    #            IMPORTING DATA TO MODEL
    #################################################

    def import_data(self, x_dir: str, y_dir: str) -> None:
        """
        Importing data from extracted features and labelled behaviours dataframes.

        Parameters
        ----------
        x_dir : str
            _description_
        y_dir : str
            _description_
        """
        out_x_dir = os.path.join(self.root_dir, X_ID)
        os.makedirs(out_x_dir, exist_ok=True)
        out_y_dir = os.path.join(self.root_dir, Y_ID)
        os.makedirs(out_y_dir, exist_ok=True)
        # Copying to model root directory
        for fp in os.listdir(x_dir):
            shutil.copyfile(os.path.join(x_dir, fp), os.path.join(out_x_dir, fp))
            shutil.copyfile(os.path.join(y_dir, fp), os.path.join(out_y_dir, fp))

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
        df_dict = {X_ID: None, Y_ID: None}
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

    @staticmethod
    def preprocess_x(x: pd.DataFrame) -> pd.DataFrame:
        """
        The preprocessing steps are:
        - MinMax scaling
        """
        # MinMax scaling X dfs
        x = pd.DataFrame(
            MinMaxScaler().fit_transform(x),
            index=x.index,
            columns=x.columns,
        )
        # Returning df
        return x

    @staticmethod
    def preprocess_y(y: pd.DataFrame) -> pd.DataFrame:
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
        # Converting MultiIndex columns to single columns
        cols_filter = np.isin(
            y.columns.get_level_values(BEHAV_CN[1]),
            [BehavColumns.PROB.value, BehavColumns.PRED.value],
            invert=True,
        )
        y = y.loc[:, cols_filter]
        # Setting the column names from `(behav, outcome)` to `{behav}__{outcome}`
        y.columns = [
            f"{i[0]}" if i[1] == BehavColumns.ACTUAL.value else f"{i[0]}__{i[1]}"
            for i in y.columns
        ]
        # Returning df
        return y

    def resample(self, y: pd.DataFrame):
        """
        Uses the resampling strategy and seed in configs.
        Returns the index for resampling. This can then be used to filter the X and y dfs.
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
        """
        # Getting array of index numbers
        resampled_arr = [df.index.get_loc(l) for l in resampled_index]
        n = self.configs.window_frames
        # Making arrays of (samples, window, features)
        return np.stack([df.iloc[i - n : i + n + 1].values for i in resampled_arr])

    def train_test_split(self, x, y):
        """
        Splitting into train and test sets
        """
        # Splitting
        # Splitting into train and test sets
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X,
        #     y,
        #     test_size=1-self.configs.train_fraction,
        #     stratify=y,
        # )
        # Manual split to separate bouts themselves
        split = int(x.shape[0] * self.configs.train_fraction)
        x_train, x_test = x[:split], x[split:]
        y_train, y_test = y[:split], y[split:]
        return x_train, x_test, y_train, y_test

    #################################################
    #            PIPELINE FOR DATA PREP
    #################################################

    def prepare_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
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
        # Preprocessing X df
        x = self.preprocess_x(x)
        # Preprocessing y df
        y = self.preprocess_y(y)
        # Selecting y class
        y = y[self.configs.behaviour_name]
        # Resampling indexes using y classes
        index = self.resample(y)
        # Making X windowed array
        x = self.make_windows(x, index)
        # Making y array
        y = y.loc[index].values.reshape(-1, 1)
        # Returning x and y
        return x, y

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

    def clf_predict(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """
        Making predictions using the given model and novel extracted features dataframe.

        Parameters
        ----------
        x : pd.DataFrame
            Novel extracted features dataframe.

        Returns
        -------
        pd.DataFrame
            Predicted behaviour classifications. Dataframe columns are in the format:
            ```
            behaviours :  behav    behav
            outcomes   :  "prob"   "pred"
            ```
        """
        # TODO: how to preprocess x (if we require a (sample, window, features) shape array)
        # Getting probabilities from model
        y_probs = self.clf.predict(x)
        y_preds = y_probs > self.configs.pcutoff
        # Making df
        index = x.index if isinstance(x, pd.DataFrame) else np.arange(x.shape[0])
        pred_df = BehavMixin.init_df(index)
        pred_df[(self.configs.behaviour_name, BehavColumns.PROB.value)] = y_probs
        pred_df[(self.configs.behaviour_name, BehavColumns.PRED.value)] = y_preds
        # Returning predicted behavs
        return pred_df

    def clf_eval(self, x, y):
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
        """
        # Making eval dir
        eval_dir = os.path.join(self.root_dir, "eval")
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
        metrics_fig = self.conf_matr_fig(y_true, y_pred)
        metrics_fig.savefig(os.path.join(eval_dir, f"{name}_confm.png"))
        # Making performance for different pcutoffs figure
        pcutoffs_fig = self.metrics_pcutoffs_fig(y_true, y_prob)
        pcutoffs_fig.savefig(os.path.join(eval_dir, f"{name}_pcutoffs.png"))
        # Logistic curve
        logc_fig = self.logistic_curve(y_true, y_prob)
        logc_fig.savefig(os.path.join(eval_dir, f"{name}_logc.png"))
        # Return evaluations
        return y_eval, metrics_fig, pcutoffs_fig, logc_fig

    def clf_pipeline(self):
        """
        Makes a classifier and saves it to the model's root directory.
        """
        # Preparing data
        x, y = self.prepare_data()
        # If using a frames classifier, then taking the middle frame of each window sample
        # NOTE: ommit for window classifiers
        x = x[:, self.configs.window_frames + 1]
        # Splitting into train and test sets
        x_train, x_test, y_train, y_test = self.train_test_split(x, y)
        # Initialising the model (cnn, dnn, or rf)
        self.init_dnn_classifier()
        # Evaluating the model (training on train, testing on test)
        self.train_nn_classifier(x_train, y_train)
        self.clf_eval(x_test, y_test)
        # Training the model (on all data)
        self.train_nn_classifier(x, y)
        self.clf_save()

    #################################################
    # EVALUATION METRICS FUNCTIONS
    #################################################

    @staticmethod
    def conf_matr_fig(y_true, y_pred):
        """
        __summary__
        """
        print(classification_report(y_true, y_pred))
        # Making confusion matrix
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(
            confusion_matrix(y_true, y_pred),
            annot=True,
            fmt="d",
            cmap="viridis",
            cbar=False,
            xticklabels=["nil", "fight"],
            yticklabels=["nil", "fight"],
            ax=ax,
        )
        return fig

    @staticmethod
    def metrics_pcutoffs_fig(y_true, y_prob):
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
            report = classification_report(y_true, y_pred, output_dict=True)
            precisions[i] = report["1"]["precision"]
            recalls[i] = report["1"]["recall"]
            f1[i] = report["1"]["f1-score"]
            accuracies[i] = report["accuracy"]
        # Making figure
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.lineplot(x=pcutoffs, y=precisions, label="precision", ax=ax)
        sns.lineplot(x=pcutoffs, y=recalls, label="recall", ax=ax)
        sns.lineplot(x=pcutoffs, y=f1, label="f1", ax=ax)
        sns.lineplot(x=pcutoffs, y=accuracies, label="accuracy", ax=ax)
        return fig

    @staticmethod
    def logistic_curve(y_true, y_prob):
        """
        __summary__
        """
        y_eval = pd.DataFrame(
            {
                "y_true": y_true,
                "y_prob": y_prob,
                "y_pred": y_prob > 0.4,
                "y_true_jitter": y_true
                + (0.2 * (np.random.rand(len(y_prob)) - 0.5)),
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

    #################################################
    # CNN CLASSIFIER
    #################################################

    def init_cnn_classifier(self):
        """
        x features is (samples, window, features).
        y outcome is (samples, class).
        """
        # Input layers
        # 546 is number of SimBA features
        inputs = Input(shape=(self.configs.window_frames * 2 + 1, 546))
        # Hidden layers
        l = Conv1D(32, 3, activation="relu")(inputs)
        l = MaxPooling1D(2)(l)
        l = Conv1D(64, 3, activation="relu")(l)
        l = MaxPooling1D(2)(l)
        l = Flatten()(l)
        l = Dense(64, activation="relu")(l)
        l = Dropout(0.5)(l)
        # Binary classification problem (probability output)
        outputs = Dense(1, activation="sigmoid")(l)
        # Create the model
        self.clf = Model(inputs=inputs, outputs=outputs)
        # Compile the model
        self.clf.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    def visualize_nn_classifier(self):
        """
        __summary__
        """
        # model.summary()
        plot_model(
            self.clf,
            to_file=os.path.join(
                self.root_dir, f"{self.configs.behaviour_name}_architecture.png"
            ),
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=200,
            show_layer_activations=True,
            show_trainable=False,
        )

    def train_nn_classifier(self, x, y):
        """
        __summary__
        """
        h = self.clf.fit(
            x,
            y,
            batch_size=64,
            epochs=200,
            validation_split=0.1,
            shuffle=True,
            verbose=1,
            callbacks=None,
        )

    #################################################
    # DNN CLASSIFIER
    #################################################

    def init_dnn_classifier(self):
        """
        x features is (samples, features).
        y outcome is (samples, class).
        """
        # Input layers
        # 546 is number of SimBA features
        input_shape = (546,)
        inputs = Input(shape=input_shape)
        # Hidden layers
        l = Dense(32, activation="relu")(inputs)  # 32, 64
        l = Dropout(0.5)(l)
        # Binary classification problem (probability output)
        outputs = Dense(1, activation="sigmoid")(l)
        # Create the model
        self.clf = Model(inputs=inputs, outputs=outputs)
        # Compiling model
        self.clf.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    #################################################
    # RF CLASSIFIER
    #################################################

    def init_rf_classifier(self):
        """
        x features is (samples, features).
        y outcome is (samples, class).
        """
        # Creating Gradient Boosting Classifier
        # self.model = GradientBoostingClassifier(
        #     n_estimators=200,
        #     learning_rate=0.1,
        #     # max_depth=3,
        #     random_state=0,
        #     verbose=1,
        # )
        self.clf = RandomForestClassifier(
            n_estimators=2000,
            max_depth=3,
            random_state=0,
            n_jobs=16,
            verbose=1,
        )
        setattr(self.clf, "predict", lambda x: self.clf.predict_proba(x)[:, 1])

    def train_sk_classifier(self, x, y):
        """
        __summary__
        """
        # Training the model
        self.clf.fit(x, y)

    #################################################
    #            COMBINING DFS TO SINGLE DF
    #################################################

    # def combine_dfs(
    #     self,
    #     x_dir: str,
    #     y_dir: str,
    # ) -> None:
    # """

    #     Concatenating the data into a single `X` df and `y` df and save to
    #     the BehavClassifier's root directory.

    #     Parameters
    #     ----------
    #     x_dir : str
    #         Directory path for extracted features dataframes.
    #     y_dir : str
    #         Directory path for labelled behaviours dataframes.

    #     Notes
    #     -----
    #     The model_config file must contain the following parameters:
    #     - todo
    #     """
    #     # Combining features and scored labels dfs together respectively
    #     x_all = self._combine_dfs(x_dir, self.configs.names_ls)
    #     y_all = self._combine_dfs(y_dir, self.configs.names_ls)
    #     # Select only rows that exist in both x_all and y_all (like an inner join)
    #     x_all = x_all[x_all.index.isin(y_all.index)]
    #     y_all = y_all[y_all.index.isin(x_all.index)]
    #     # Checking x and y dfs
    #     BehavClassifier.check_comb_x_df(x_all)
    #     BehavClassifier.check_comb_y_df(y_all)
    #     # Sorting index and saving to output
    #     DFIOMixin.write_feather(x_all.sort_index(), self.get_df_fp("x_all"))
    #     DFIOMixin.write_feather(y_all.sort_index(), self.get_df_fp("y_all"))

    # def _combine_dfs(self, in_dir: str, names_ls: list[str]) -> pd.DataFrame:
    #     """
    #     Combine a list of dataframes into a single dataframe.
    #     The experiment ID is added as a level to the index.
    #     """
    #     return pd.concat(
    #         [
    #             pd.concat(
    #                 [DFIOMixin.read_feather(os.path.join(in_dir, f"{name}.feather"))],
    #                 keys=[name],
    #                 names=["experiments"],
    #                 axis=0,
    #             )
    #             for name in names_ls
    #         ],
    #         axis=0,
    #     )

    # #################################################
    # #         MAKING TRAIN TEST SPLITS
    # #################################################

    # def make_train_test_split(self):
    #     """
    #     Making train and test splits from _all dfs (i.e. the combined dfs)
    #     """
    #     # Making train/test fraction lists in configs json
    #     self._init_train_test_split()
    #     # Loading _all dfs
    #     x_all = BehavClassifier.read_comb_x_feather(self.get_df_fp("x_all"))
    #     y_all = BehavClassifier.read_comb_y_feather(self.get_df_fp("y_all"))
    #     # Making train/test split
    #     x_train, x_test = self._make_train_test_split(x_all)
    #     y_train, y_test = self._make_train_test_split(y_all)
    #     # Sorting index and saving each train/test file
    #     DFIOMixin.write_feather(x_train.sort_index(), self.get_df_fp("x_train"))
    #     DFIOMixin.write_feather(x_test.sort_index(), self.get_df_fp("x_test"))
    #     DFIOMixin.write_feather(y_train.sort_index(), self.get_df_fp("y_train"))
    #     DFIOMixin.write_feather(y_test.sort_index(), self.get_df_fp("y_test"))

    # def _init_train_test_split(self):
    #     """Making train and test split experiments lists in the configs."""
    #     # Splitting videos into training and test sets
    #     train_n = int(self.configs.train_fraction * len(self.configs.all_ls))
    #     # Saving train and test experiments lists
    #     configs = self.configs
    #     # Selecting random train set
    #     configs.train_ls = np.random.choice(
    #         a=self.configs.all_ls, size=train_n, replace=False
    #     ).tolist()
    #     # Everything else is the test set
    #     configs.test_ls = np.array(self.configs.all_ls)[
    #         ~np.isin(self.configs.all_ls, self.configs.train_ls)
    #     ].tolist()
    #     configs.write_json(self.configs_fp)

    # def _make_train_test_split(
    #     self, all_df: pd.DataFrame
    # ) -> tuple[pd.DataFrame, pd.DataFrame]:
    #     """Make the combined train and test dataframes from the combined all dataframe."""
    #     # Create and save _train dfs
    #     train_df = all_df[all_df.index.isin(self.configs.train_ls, level="experiments")]
    #     test_df = all_df[all_df.index.isin(self.configs.test_ls, level="experiments")]
    #     return train_df, test_df

    # #################################################
    # #         RANDOM UNDERSAMPLING
    # #################################################

    # def make_random_undersample(self):
    #     """
    #     Performing random undersampling for given behaviour.

    #     Notes
    #     -----
    #     The model_config file must contain the following parameters:
    #     - name: str
    #     - undersampling_strategy: float
    #     - seed: int
    #     """
    #     # For each "all" and "train" datasets
    #     for i in [Datasets.ALL, Datasets.TRAIN]:
    #         i = i.value
    #         # Reading in df
    #         x_df = BehavClassifier.read_comb_x_feather(self.get_df_fp(f"x_{i}"))
    #         y_df = BehavClassifier.read_comb_y_feather(self.get_df_fp(f"y_{i}"))
    #         # Preparing ID index to subsample on. These will store the index numbers subsampled on
    #         index = y_df.index.to_list()  # TODO: try ".values"
    #         # Preparing y_vals to subsample on. These will store the y values as a 1D array
    #         y_vals = y_df[(self.configs.name, BehavColumns.ACTUAL.value)].values
    #         # Random under-subsampling (returns subsampled index IDs)
    #         undersampler = RandomUnderSampler(
    #             sampling_strategy=self.configs.undersampling_strategy,
    #             random_state=self.configs.seed,
    #         )
    #         index, _ = undersampler.fit_resample(index, y_vals)
    #         # Formatting index_subs IDs so they match indexes in _dfs
    #         index = [(i[0], int(i[1])) for i in index]
    #         # Filtering x_df and y_dfs for selected undersampled index IDs
    #         x_df = x_df.loc[index]
    #         y_df = y_df.loc[index]
    #         # Selecting subsampled index ID rows, sorting, and saving
    #         DFIOMixin.write_feather(x_df.sort_index(), self.get_df_fp(f"x_{i}_subs"))
    #         DFIOMixin.write_feather(y_df.sort_index(), self.get_df_fp(f"y_{i}_subs"))

    # #################################################
    # #       MAKE, TRAIN, RUN SCIKIT CLASSIFIER
    # #################################################

    # def init_behav_classifier(self):
    #     """
    #     Save hyper-params for the classfier (i.e. a blueprint) to the configs json for
    #     given behaviour.

    #     Notes
    #     -----
    #     The model_config file must contain the following parameters:
    #     ```
    #     - seed: int
    #     ```
    #     """
    #     # Initialising and defining model
    #     model = GradientBoostingClassifier(
    #         n_estimators=2000,
    #         learning_rate=0.1,
    #         loss="log_loss",
    #         criterion="friedman_mse",
    #         max_features="sqrt",
    #         random_state=self.configs.seed,
    #         subsample=1.0,
    #         verbose=1,
    #         # njobs=-1,
    #     )
    #     # Saving model hyper parameters to configs file
    #     configs = self.configs
    #     configs.model_type = str(type(model))
    #     configs.model_params = model.get_params()
    #     configs.write_json(self.configs_fp)

    # def train_behav_classifier(self, dataset: Datasets = Datasets.ALL):
    #     """
    #     Making classifier from configs json blueprint and training it on `all` and `_train` data.
    #     Saving a separate trained classifier for each dataset (`all` and `train`).
    #     """
    #     dataset = Datasets(dataset).value
    #     # Training model on all data
    #     # Loading in X/y dfs
    #     x_df = BehavClassifier.read_comb_x_feather(self.get_df_fp(f"x_{dataset}_subs"))
    #     y_df = BehavClassifier.read_comb_y_feather(self.get_df_fp(f"y_{dataset}_subs"))
    #     y_vals = y_df[(self.configs.name, BehavColumns.ACTUAL.value)].values
    #     # Making model
    #     model = GradientBoostingClassifier(**self.configs.model_params)
    #     # Training model
    #     model.fit(x_df, y_vals)
    #     # Saving model
    #     joblib.dump(model, self.get_model_fp(f"model_{dataset}"))

    # #################################################
    # #         RUN MODEL PREDICTIONS
    # #################################################

    # def model_predict(
    #     self,
    #     x_df: pd.DataFrame,
    #     model_dataset: Datasets = Datasets.ALL,
    # ) -> pd.DataFrame:
    #     """
    #     Making predictions using the given model and novel extracted features dataframe.
    #     The default model used for the given BehavClassifier is `Datasets.ALL`.

    #     Parameters
    #     ----------
    #     x_df : pd.DataFrame
    #         Novel extracted features dataframe.
    #     dataset : str, optional
    #         Model name, by default `Datasets.ALL`.

    #     Returns
    #     -------
    #     pd.DataFrame
    #         Predicted behaviour classifications. Dataframe columns are in the format:
    #         ```
    #         behaviours :  behav    behav
    #         outcomes   :  "prob"   "pred"
    #         ```
    #     """
    #     model_dataset = Datasets(model_dataset).value
    #     # Loading in each model
    #     model = joblib.load(self.get_model_fp(f"model_{model_dataset}"))
    #     # Getting probabilities from model
    #     probs = model.predict_proba(x_df)[:, 1]
    #     preds = (probs > self.configs.pcutoff).astype(np.uint8)
    #     # Making df
    #     preds_df = BehavMixin.init_df(x_df.index)
    #     preds_df[(self.configs.name, BehavColumns.PROB.value)] = probs
    #     preds_df[(self.configs.name, BehavColumns.PRED.value)] = preds
    #     # Returning predicted behavs
    #     return preds_df

    # #################################################
    # #      EVALUATE MODEL WITH TRAIN/TEST DATA
    # #################################################

    # def model_eval(self, dataset: Datasets = Datasets.TRAIN):
    #     """
    #     Evaluating the model using the model trained on the _train data.

    #     Notes
    #     -----
    #     The model_config file must contain the following parameters:
    #     ```
    #     - name: str
    #     ```
    #     """
    #     dataset = Datasets(dataset).value
    #     # Loading test X data
    #     x_test = BehavClassifier.read_comb_x_feather(
    #         self.get_df_fp(f"x_{Datasets.TEST.value}")
    #     )
    #     # Getting model predictions for evaluation
    #     eval_df = self.model_predict(x_test, dataset)
    #     # Adding actual y labels
    #     y_test = BehavClassifier.read_comb_y_feather(
    #         self.get_df_fp(f"y_{Datasets.TEST.value}")
    #     )
    #     eval_df[(self.configs.name, BehavColumns.ACTUAL.value)] = y_test[
    #         (self.configs.name, BehavColumns.ACTUAL.value)
    #     ].values
    #     # Saving eval df to file
    #     eval_fp = os.path.join(self.root_dir, "eval", "eval_df.feather")
    #     DFIOMixin.write_feather(eval_df, eval_fp)
    #     # Making pcutoff metrics plot
    #     fig = self.model_eval_plot_metrics(eval_df)
    #     fig.savefig(os.path.join(self.root_dir, "eval", "pcutoff_metrics.png"))
    #     fig.clf()
    #     # Making logistic results plot
    #     fig = self.model_eval_plot_results(eval_df)
    #     fig.savefig(os.path.join(self.root_dir, "eval", "logistic_results.png"))
    #     fig.clf()

    # #################################################
    # #         EVALUATE MODEL PREDICTIONS
    # #################################################

    # def model_eval_report(self, eval_df: pd.DataFrame) -> None:
    #     """
    #     Printing model evaluation summaries.

    #     Parameters
    #     ----------
    #     eval_df : pd.DataFrame
    #         fbf evaluation dataframe.
    #     """
    #     # Printing eval report
    #     print(
    #         confusion_matrix(
    #             eval_df[(self.configs.name, BehavColumns.ACTUAL.value)],
    #             eval_df[(self.configs.name, BehavColumns.PRED.value)],
    #             labels=[0, 1],
    #         )
    #     )
    #     print(
    #         classification_report(
    #             eval_df[(self.configs.name, BehavColumns.ACTUAL.value)],
    #             eval_df[(self.configs.name, BehavColumns.PRED.value)],
    #         )
    #     )

    # def model_eval_plot_timeseries(self, eval_df: pd.DataFrame) -> Figure:
    #     """
    #     Plots timeseries of behav probability against actual behaviour
    #     """
    #     # Getting predictions eval df
    #     eval_long_df = (
    #         eval_df[
    #             [
    #                 (self.configs.name, BehavColumns.ACTUAL.value),
    #                 (self.configs.name, BehavColumns.PROB.value),
    #             ]
    #         ]
    #         .reset_index(names="frame")
    #         .melt(id_vars=["frame"], var_name="measure", value_name="value")
    #     )
    #     # Plotting each outcome (through time)
    #     fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    #     sns.lineplot(
    #         data=eval_long_df,
    #         x="frame",
    #         y="value",
    #         hue="measure",
    #         palette="rainbow",
    #         alpha=0.6,
    #         ax=ax,
    #     )
    #     ax.set_ylim(-0.01, 1.01)
    #     return fig

    # def model_eval_plot_results(self, eval_df: pd.DataFrame) -> Figure:
    #     """
    #     Plotting outcome against ML probability (sorted by ML probability)
    #     """
    #     # Sorting eval_df df by y_prob
    #     eval_df = eval_df[self.configs.name]
    #     eval_df = eval_df.sort_values(
    #         (BehavColumns.PROB.value), ignore_index=True
    #     ).reset_index()
    #     # Making plot with actual outcomes (scatter) and ML probaility outcomes (line)
    #     fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    #     eval_df[f"{BehavColumns.ACTUAL.value}_jitter"] = (
    #         eval_df[BehavColumns.ACTUAL.value]
    #         + (np.random.rand(eval_df.shape[0]) - 0.5) * 0.1
    #     )

    #     sns.scatterplot(
    #         data=eval_df,
    #         x="prob",
    #         y=f"{BehavColumns.ACTUAL.value}_jitter",
    #         hue=BehavColumns.PRED.value,
    #         marker=".",
    #         s=10,
    #         linewidth=0,
    #         alpha=0.2,
    #         ax=ax,
    #     )
    #     # Making axis titles
    #     ax.set_title("Logistic outcomes against ML probability")
    #     ax.set_xlabel("Sample")
    #     ax.set_ylabel("ML probability")
    #     ax.set_ylim(-0.11, 1.11)
    #     ax.set_xticks([])
    #     return fig

    # def model_eval_plot_metrics(self, eval_df: pd.DataFrame) -> Figure:
    #     """
    #     PLOTTING RECALL, PRECISION, F1, AND ACCURACY FOR DIFFERENT PCUTOFFS.
    #     """
    #     beta = 1.5
    #     pcutoffs = np.linspace(0, 1, 101)[:-1]
    #     eval_df = eval_df[self.configs.name]
    #     # Initialising df_eval_pcutoffs df
    #     df_eval_pcutoffs = pd.DataFrame()
    #     for pcutoff in pcutoffs:
    #         # Getting y_true and y_pred
    #         y_true = eval_df[BehavColumns.ACTUAL.value]
    #         y_pred = (eval_df[BehavColumns.PROB.value] >= pcutoff).astype(int)
    #         # Getting performance metrics
    #         precision, recall, fscore, _ = precision_recall_fscore_support(
    #             y_true, y_pred, labels=[0, 1], beta=beta
    #         )
    #         accuracy = accuracy_score(y_true, y_pred)
    #         # Getting eval metrics
    #         df_eval_pcutoffs.loc[pcutoff, "precision"] = precision[1]
    #         df_eval_pcutoffs.loc[pcutoff, "recall"] = recall[1]
    #         df_eval_pcutoffs.loc[pcutoff, "fscore"] = fscore[1]
    #         df_eval_pcutoffs.loc[pcutoff, "accuracy"] = accuracy
    #     # Plotting performance metrics for different pcutoff values
    #     df_eval_pcutoffs_long = (
    #         df_eval_pcutoffs.reset_index(names="pcutoff")
    #         .melt(id_vars=["pcutoff"], var_name="measure", value_name="value")
    #         .infer_objects()
    #     )
    #     fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    #     sns.lineplot(
    #         data=df_eval_pcutoffs_long,
    #         x="pcutoff",
    #         y="value",
    #         hue="measure",
    #         palette="Pastel1",
    #         ax=ax,
    #     )
    #     # Making axis titles
    #     ax.set_title("Evaluation of behav detection")
    #     ax.set_xlabel("pcutoff")
    #     ax.set_ylabel("Metric")
    #     ax.set_ylim(0, 1.01)
    #     ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    #     return fig
    #     return fig
    #     return fig
    #     return fig
    #     return fig
