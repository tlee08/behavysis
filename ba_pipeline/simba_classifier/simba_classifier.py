"""
_summary_
"""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ba_core.mixins.df_io_mixin import DFIOMixin
from ba_core.utils.constants import (
    BEHAV_ACTUAL_COL,
    BEHAV_COLUMN_NAMES,
    BEHAV_PRED_COL,
    BEHAV_PROB_COL,
)
from imblearn.under_sampling import RandomUnderSampler
from matplotlib.figure import Figure
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ba_pipeline.simba_classifier.simba_classifier_configs import SimbaClassifierConfigs

if TYPE_CHECKING:
    from ba_pipeline.pipeline.project import BAProject


class SimbaClassifier:
    """
    SimbaClassifier class peforms SimBA model preparation, training, saving, evaluation,
    and inference.
    """

    def __init__(
        self,
        configs_fp: str,
        configs: Optional[SimbaClassifierConfigs] = None,
    ) -> None:
        """
        Make a SimbaClassifier instance.

        Parameters
        ----------
        configs_fp : str
            _description_
        configs : SimbaClassifierConfigs, optional
            _description_, by default SimbaClassifierConfigs()
        """
        # Making configs json
        self.configs_fp = configs_fp
        # Trying to read in configs json
        if configs is None:
            try:
                configs = SimbaClassifierConfigs.read_json(self.configs_fp)
            except FileNotFoundError:
                configs = SimbaClassifierConfigs()
        # Saving configs
        configs.write_json(self.configs_fp)

    @classmethod
    def from_baproject(cls, proj: BAProject) -> SimbaClassifier:
        """
        Loading classifier from given BAProject instance.

        Parameters
        ----------
        proj : BAProject
            The BAProject instance.

        Returns
        -------
        SimbaClassifier
            The loaded SimbaClassifier instance.
        """
        configs_fp: str = os.path.abspath(
            os.path.join(proj.root_dir, "behav_models", "model_configs.json")
        )
        configs: SimbaClassifierConfigs = SimbaClassifierConfigs(
            all_ls=[i.name for i in proj.get_experiments()]
        )
        return cls(configs_fp, configs)

    @property
    def configs(self) -> SimbaClassifierConfigs:
        """Returns the config model from the expected config file."""
        return SimbaClassifierConfigs.read_json(self.configs_fp)

    @property
    def root_dir(self) -> str:
        """Returns the model's root directory"""
        return os.path.split(self.configs_fp)[0]

    def make_behav_model_subdir(self, behav: str):
        """
        Making model subdir for given behaviour.
        Returns the model_subdir's instance.

        Helpful to select only the y dataframe columns with the specific behaviour
        to make a classifier for *that* particular behaviour.

        Parameters
        ----------
        behav : str
            Behaviour name.

        Returns
        -------
        SimbaClassifier
            Created SimbaClassifier instance of subdirectory behav.
        """
        # Checking if behav is in y_all
        self.check_behav_exists(behav)
        # Getting necessary config params
        behav_dir = os.path.join(self.root_dir, behav)
        os.makedirs(behav_dir, exist_ok=True)
        # Copying and saving configs JSON, and returning instance of new subdir
        configs_fp = os.path.join(self.root_dir, behav, "model_configs.json")
        # Modifying configs for new behaviour
        configs = self.configs
        configs.name = behav
        behav_clf = SimbaClassifier(configs_fp, configs)
        # For each type of X/y dataframe
        for i in ["all", "train", "test"]:
            # Copying X dataframes
            shutil.copyfile(self.get_df_fp(f"x_{i}"), behav_clf.get_df_fp(f"x_{i}"))
            # Selecing y dataframe behav cols and saving to model behav dir
            y_df = DFIOMixin.read_feather(self.get_df_fp(f"y_{i}"))
            y_df = y_df.loc[:, behav]
            DFIOMixin.write_feather(y_df, behav_clf.get_df_fp(f"y_{i}"))
        # Returning SimbaClassfier instance of new subdir
        return behav_clf

    #################################################
    #            GETTER AND SETTERS
    #################################################

    def get_model_fp(self, model_name: str) -> str:
        """
        Returns the model fp (absolute fp), with the following format:
        ```
        "<model_root_dir>/<model_name>.sav"
        ```

        Parameters
        ----------
        model_name : str
            model_name.

        Returns
        -------
        str
            model .sav filepath.
        """
        return os.path.join(self.root_dir, f"{model_name}.sav")

    def get_df_fp(self, df_name: str) -> str:
        """
        Returns the dataframe fp (absolute fp), with the following format:
        ```
        "<model_root_dir>/<df_name>.feather"
        ```

        Parameters
        ----------
        df_name : str
            df_name.

        Returns
        -------
        str
            dataframe .feather filepath.
        """
        return os.path.join(self.root_dir, f"{df_name}.feather")

    def check_behav_exists(self, behav: str) -> None:
        """
        Checks if the given behaviour string exists as a behaviour label
        in the `y_all` dataframe.
        """
        y_all = DFIOMixin.read_feather(self.get_df_fp("y_all"))
        behavs_ls = y_all.columns.unique("behaviours")
        if behav not in behavs_ls:
            raise ValueError(f"{behav} is not in the `y_all` dataframe.")

    #################################################
    #            COMBINING DFS TO SINGLE DF
    #################################################

    def combine_dfs(
        self,
        x_dir: str,
        y_dir: str,
    ) -> None:
        """
        Concatenating the data into a single `X` df and `y` df and save to
        the SimbaClassifier's root directory.

        Parameters
        ----------
        x_dir : str
            Directory path for extracted features dataframes.
        y_dir : str
            Directory path for labelled behaviours dataframes.

        Notes
        -----
        The model_config file must contain the following parameters:
        - all_ls: list
        """
        # Combining features and scored labels dfs together respectively
        x_all = self._combine_dfs(x_dir, self.configs.all_ls)
        y_all = self._combine_dfs(y_dir, self.configs.all_ls)
        # Select only rows that exist in both x_all and y_all (like an inner join)
        x_all = x_all[x_all.index.isin(y_all.index)]
        y_all = y_all[y_all.index.isin(x_all.index)]
        # Sorting index and saving to output
        DFIOMixin.write_feather(x_all.sort_index(), self.get_df_fp("x_all"))
        DFIOMixin.write_feather(y_all.sort_index(), self.get_df_fp("y_all"))

    def _combine_dfs(self, in_dir: str, names_ls: list[str]) -> pd.DataFrame:
        """
        Combine a list of dataframes into a single dataframe.
        The experiment ID is added as a level to the index.
        """
        return pd.concat(
            [
                self._combine_dfs_worker(os.path.join(in_dir, f"{i}.feather"), i)
                for i in names_ls
            ],
            axis=0,
        )

    def _combine_dfs_worker(self, fp: str, name: str) -> pd.DataFrame:
        """Add the name of the experiment to the index of the dataframe."""
        return pd.concat(
            [DFIOMixin.read_feather(fp)], keys=[name], names=["experiments"], axis=0
        )

    #################################################
    #         MAKING TRAIN TEST SPLITS
    #################################################

    def make_train_test_split(self):
        """
        Making train and test splits from _all dfs (i.e. the combined dfs)
        """
        # Making train/test fraction lists in configs json
        self._init_train_test_split()
        # Loading _all dfs
        x_all = DFIOMixin.read_feather(self.get_df_fp("x_all"))
        y_all = DFIOMixin.read_feather(self.get_df_fp("y_all"))
        # Making train/test split
        x_train, x_test = self._make_train_test_split(x_all)
        y_train, y_test = self._make_train_test_split(y_all)
        # Sorting index and saving each train/test file
        DFIOMixin.write_feather(x_train.sort_index(), self.get_df_fp("x_train"))
        DFIOMixin.write_feather(x_test.sort_index(), self.get_df_fp("x_test"))
        DFIOMixin.write_feather(y_train.sort_index(), self.get_df_fp("y_train"))
        DFIOMixin.write_feather(y_test.sort_index(), self.get_df_fp("y_test"))

    def _init_train_test_split(self):
        """Making train and test split experiments lists in the configs."""
        # Splitting videos into training and test sets
        train_n = int(self.configs.train_fraction * len(self.configs.all_ls))
        # Saving train and test experiments lists
        configs = self.configs
        configs.train_ls = np.random.choice(
            a=self.configs.all_ls, size=train_n, replace=False
        ).tolist()
        configs.test_ls = np.array(self.configs.all_ls)[
            ~np.isin(self.configs.all_ls, self.configs.train_ls)
        ].tolist()
        configs.write_json(self.configs_fp)

    def _make_train_test_split(
        self, all_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Make the combined train and test dataframes from the combined all dataframe."""
        # Create and save _train dfs
        train_df = all_df[all_df.index.isin(self.configs.train_ls, level="experiments")]
        test_df = all_df[all_df.index.isin(self.configs.test_ls, level="experiments")]
        return train_df, test_df

    #################################################
    #         RANDOM UNDERSAMPLING
    #################################################

    def make_random_ovundersample(self):
        """
        Making random oversampler then undersampling for given behaviour.

        TODO: rationale is to balance the classes for the classifier, and select
        more examples of majority frames when they are randomly sampled.

        Notes
        -----
        The model_config file must contain the following parameters:
        - name: str
        - undersampling_strategy: float
        - seed: int
        """
        # For each "all" and "train" datasets
        for i in ["all", "train"]:
            # Reading in df
            x_df = pd.read_feather(self.get_df_fp(f"x_{i}"))
            y_df = pd.read_feather(self.get_df_fp(f"y_{i}"))
            # Preparing ID index to subsample on. These will store the index numbers subsampled on
            index = y_df.index.to_list()  # TODO: try ".values"
            # Preparing y_vals to subsample on. These will store the y values as a 1D array
            y_vals = y_df[(self.configs.name, BEHAV_ACTUAL_COL)].values
            # Random under-subsampling (returns subsampled index IDs)
            undersampler = RandomUnderSampler(
                sampling_strategy=self.configs.undersampling_strategy,
                random_state=self.configs.seed,
            )
            index, _ = undersampler.fit_resample(index, y_vals)
            # Formatting index_subs IDs so they match indexes in _dfs
            index = [(i[0], int(i[1])) for i in index]
            # Filtering x_df and y_dfs for selected undersampled index IDs
            x_df = x_df.loc[index]
            y_df = y_df.loc[index]
            # Selecting subsampled index ID rows, sorting, and saving
            DFIOMixin.write_feather(x_df.sort_index(), self.get_df_fp(f"x_{i}_subs"))
            DFIOMixin.write_feather(y_df.sort_index(), self.get_df_fp(f"y_{i}_subs"))

    #################################################
    #       MAKE, TRAIN, RUN SCIKIT CLASSIFIER
    #################################################

    def init_behav_classifier(self):
        """
        Save hyper-params for the classfier (i.e. a blueprint) to the configs json for
        given behaviour.

        Notes
        -----
        The model_config file must contain the following parameters:
        ```
        - seed: int
        ```
        """
        # Initialising and defining model
        model = GradientBoostingClassifier(
            n_estimators=2000,
            learning_rate=0.1,
            loss="log_loss",
            criterion="friedman_mse",
            max_features="sqrt",
            random_state=self.configs.seed,
            subsample=1.0,
            verbose=1,
            # njobs=-1,
        )
        # Saving model hyper parameters to configs file
        configs = self.configs
        configs.model_type = str(type(model))
        configs.model_params = model.get_params()
        configs.write_json(self.configs_fp)

    def train_behav_classifier(self):
        """
        Making classifier from configs json blueprint and training it on `_all` and `_train` data.
        Saving a separate trained classifier for each dataset (`_all` and `_train`).
        """
        # Training model on _all data
        for i in ["all", "train"]:
            # Loading in X/y dfs
            x_df = DFIOMixin.read_feather(self.get_df_fp(f"x_{i}_subs"))
            y_df = DFIOMixin.read_feather(self.get_df_fp(f"y_{i}_subs"))
            y_vals = y_df[(self.configs.name, BEHAV_ACTUAL_COL)].values
            # Making model
            model = GradientBoostingClassifier(**self.configs.model_params)
            # Training model
            model.fit(x_df, y_vals)
            # Saving model
            joblib.dump(model, self.get_model_fp(f"model_{i}"))

    #################################################
    #         RUN MODEL PREDICTIONS
    #################################################

    def model_predict(
        self,
        x_df: pd.DataFrame,
        model_name: str = "model_all",
    ) -> pd.DataFrame:
        """
        Making predictionsusing the given model and novel extracted features dataframe.
        The default model used for the given SimbaClassifier is `"model_all"`.

        Parameters
        ----------
        x_df : pd.DataFrame
            Novel extracted features dataframe.
        model_name : {"model_all", "model_train"}, optional
            model name, by default "model_all".

        Returns
        -------
        pd.DataFrame
            Predicted behaviour classifications. Dataframe columns are in the format:
            ```
            behaviours :  behav    behav
            outcomes   :  "prob"   "pred"
            ```
        """
        # Loading in each model
        model = joblib.load(self.get_model_fp(model_name))
        # Getting probabilitites from model
        probs = model.predict_proba(x_df)[:, 1]
        preds = (probs > self.configs.pcutoff).astype(np.uint8)
        # Making df
        behav_preds = pd.DataFrame(index=x_df.index)
        behav_preds[(self.configs.name, BEHAV_PROB_COL)] = probs
        behav_preds[(self.configs.name, BEHAV_PRED_COL)] = preds
        # Converting columns to MultiIndex
        behav_preds.columns = pd.MultiIndex.from_tuples(
            behav_preds.columns, names=BEHAV_COLUMN_NAMES
        )
        # Returning predicted behavs
        return behav_preds

    #################################################
    #      EVALUATE MODEL WITH TRAIN/TEST DATA
    #################################################

    def model_eval(self):
        """
        Evaluating the model using the model trained on the _train data.

        Notes
        -----
        The model_config file must contain the following parameters:
        ```
        - name: str
        ```
        """
        # Loading test X data
        x_test = DFIOMixin.read_feather(self.get_df_fp("x_test"))
        # Getting model predictions for evaluation
        eval_df = self.model_predict(x_test, model_name="model_train")
        # Adding actual y labels
        y_test = DFIOMixin.read_feather(self.get_df_fp("y_test"))
        eval_df[(self.configs.name, BEHAV_ACTUAL_COL)] = y_test[
            (self.configs.name, BEHAV_ACTUAL_COL)
        ].values
        # Saving eval df to file
        eval_fp = os.path.join(self.root_dir, "eval", "eval_df.feather")
        DFIOMixin.write_feather(eval_df, eval_fp)
        # Making pcutoff metrics plot
        fig = self.model_eval_plot_metrics(eval_df)
        fig.savefig(os.path.join(self.root_dir, "eval", "pcutoff_metrics.png"))
        # Making logistic results plot
        fig = self.model_eval_plot_results(eval_df)
        fig.savefig(os.path.join(self.root_dir, "eval", "logistic_results.png"))

    #################################################
    #         EVALUATE MODEL PREDICTIONS
    #################################################

    def model_eval_report(self, eval_df: pd.DataFrame) -> None:
        """
        Printing model evaluation summaries.

        Parameters
        ----------
        eval_df : pd.DataFrame
            fbf evaluation dataframe.
        """
        # Printing eval report
        print(
            confusion_matrix(
                eval_df[BEHAV_ACTUAL_COL], eval_df[BEHAV_PRED_COL], labels=[0, 1]
            )
        )
        print(classification_report(eval_df[BEHAV_ACTUAL_COL], eval_df[BEHAV_PRED_COL]))

    def model_eval_plot_timeseries(self, eval_df: pd.DataFrame) -> Figure:
        """
        Plots timeseries of behav probability against actual behaviour
        """
        # Getting predictions eval df
        eval_long_df = (
            eval_df[[BEHAV_ACTUAL_COL, BEHAV_PROB_COL]]
            .reset_index(names="frame")
            .melt(id_vars=["frame"], var_name="measure", value_name="value")
        )
        # Plotting each outcome (through time)
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        sns.lineplot(
            data=eval_long_df,
            x="frame",
            y="value",
            hue="measure",
            palette="rainbow",
            alpha=0.6,
            ax=ax,
        )
        ax.set_ylim(-0.01, 1.01)
        return fig

    def model_eval_plot_results(self, eval_df: pd.DataFrame) -> Figure:
        """
        Plotting outcome against ML probability (sorted by ML probability)
        """
        # Sorting eval_df df by y_prob
        eval_df = eval_df.sort_values(BEHAV_PROB_COL, ignore_index=True).reset_index()
        # Making plot with actual outcomes (scatter) and ML probaility outcomes (line)
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        sns.scatterplot(
            data=eval_df.assign(
                y_true=eval_df[BEHAV_ACTUAL_COL]
                + (np.random.rand(eval_df.shape[0]) - 0.5) * 0.1
            ),
            x="index",
            y=BEHAV_ACTUAL_COL,
            c="red",
            marker=".",
            s=10,
            linewidth=0,
            alpha=0.2,
            ax=ax,
        )
        sns.lineplot(
            data=eval_df,
            x="index",
            y=BEHAV_PROB_COL,
            ax=ax,
        )
        # Making axis titles
        ax.set_title("Logistic outcomes against ML probability")
        ax.set_xlabel("Sample")
        ax.set_ylabel("ML probability")
        ax.set_ylim(-0.11, 1.11)
        ax.set_xticks([])
        return fig

    def model_eval_plot_metrics(self, eval_df: pd.DataFrame) -> Figure:
        """
        PLOTTING RECALL, PRECISION, F1, AND ACCURACY FOR DIFFERENT PCUTOFFS.
        """
        beta = 1.5
        pcutoffs = np.linspace(0, 1, 101)[:-1]
        # Initialising df_eval_pcutoffs df
        df_eval_pcutoffs = pd.DataFrame()
        for pcutoff in pcutoffs:
            # Getting y_true and y_pred
            y_true = eval_df[BEHAV_ACTUAL_COL]
            y_pred = (eval_df[BEHAV_PROB_COL] >= pcutoff).astype(int)
            # Getting performance metrics
            precision, recall, fscore, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=[0, 1], beta=beta
            )
            accuracy = accuracy_score(y_true, y_pred)
            # Getting eval metrics
            df_eval_pcutoffs.loc[pcutoff, "precision"] = precision[1]
            df_eval_pcutoffs.loc[pcutoff, "recall"] = recall[1]
            df_eval_pcutoffs.loc[pcutoff, "fscore"] = fscore[1]
            df_eval_pcutoffs.loc[pcutoff, "accuracy"] = accuracy
        # Plotting performance metrics for different pcutoff values
        df_eval_pcutoffs_long = (
            df_eval_pcutoffs.reset_index(names="pcutoff")
            .melt(id_vars=["pcutoff"], var_name="measure", value_name="value")
            .infer_objects()
        )
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        sns.lineplot(
            data=df_eval_pcutoffs_long,
            x="pcutoff",
            y="value",
            hue="measure",
            palette="Pastel1",
            ax=ax,
        )
        # Making axis titles
        ax.set_title("Evaluation of behav detection")
        ax.set_xlabel("pcutoff")
        ax.set_ylabel("Metric")
        ax.set_ylim(0, 1.01)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        return fig


#     @staticmethod
#     def compare_models(
#         model_dir_ls: str,
#         X: pd.DataFrame,
#         y_true: pd.DataFrame,
#         pcutoff: float,
#     ) -> pd.DataFrame:
#         """
#         Comparing different models.
#         """
#         beta = 1.5
#         df_eval = pd.DataFrame()
#         for model_dir_i in model_dir_ls:
#             # Getting y_true and y_pred
#             eval_df = model_predict(model_dir_i, X, y_true, pcutoff)
#             y_true = eval_df[BEHAV_ACTUAL_COL]
#             y_pred = (eval_df[BEHAV_PROB_COL] >= pcutoff).astype(int)
#             # Getting performance metrics
#             precision, recall, fscore, support = precision_recall_fscore_support(
#                 y_true, y_pred, labels=[0, 1], beta=beta
#             )
#             accuracy = accuracy_score(y_true, y_pred)
#             # Getting eval metrics
#             df_eval.loc[model_dir_i, "precision"] = precision[1]
#             df_eval.loc[model_dir_i, "recall"] = recall[1]
#             df_eval.loc[model_dir_i, "fscore"] = fscore[1]
#             df_eval.loc[model_dir_i, "accuracy"] = accuracy
#         # Returning eval
#         return df_eval

#     @staticmethod
#     def model_eval_bad_bouts(eval_df: pd.DataFrame) -> pd.DataFrame:
#         # Making unique fight ID for each bout (by sequences of BEHAV_ACTUAL_COL and "exp")
#         if "exp" in eval_df.columns:
#             eval_df["bout_id"] = (
#                 (eval_df[BEHAV_ACTUAL_COL] != eval_df[BEHAV_ACTUAL_COL].shift())
#                 | (eval_df["exp"] != eval_df["exp"].shift())
#             ).cumsum()
#         else:
#             eval_df["bout_id"] = (
#                 (eval_df[BEHAV_ACTUAL_COL] != eval_df[BEHAV_ACTUAL_COL].shift())
#             ).cumsum()
#         # Grouping by fight ID and getting proportion of pred fight
#         eval_bouts_df = eval_df.groupby(["exp", "bout_id"]).agg(
#             {
#                 BEHAV_PRED_COL: ["count", np.sum, np.mean],
#                 BEHAV_ACTUAL_COL: [np.mean],
#             }
#         )
#         # Calculating error = proportion of incorrectly predicted behaviour for eahc bout
#         eval_bouts_df["error"] = np.abs(
#             eval_bouts_df[(BEHAV_PRED_COL, "mean")]
#             - eval_bouts_df[(BEHAV_ACTUAL_COL, "mean")]
#         )
#         # Returning error of bouts
#         return eval_bouts_df

#     @staticmethod
#     def compare_models_plot_metrics(df_eval: pd.DataFrame) -> Figure:
#         # Plotting performance metrics across different models
#         df_eval_long = (
#             df_eval.assign(model_id=pd.factorize(df_eval.index)[0].astype(str))
#             .melt(id_vars=["model_id"], var_name="measure", value_name="value")
#             .infer_objects()
#         )
#         fig, ax = plt.subplots(figsize=(16, 10), layout="constrained")
#         sns.barplot(
#             data=df_eval_long,
#             x="measure",
#             y="value",
#             hue="model_id",
#             palette="Pastel1",
#             ax=ax,
#         )
#         # Making axis titles
#         ax.set_title("Comparison of ML models")
#         ax.set_xlabel("Metric")
#         ax.set_ylabel("Value")
#         ax.set_ylim(0, 1.01)
#         ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), title="model id")
#         # Annotating bars with values
#         for container in ax.containers:
#             ax.bar_label(container, fmt="{:.3f}")
#         return fig


# #################################################
# #         MAKE, TRAIN, RUN KERAS CLASSIFIER
# #################################################


# # def model_keras_to_scikit(model, hparams={}):
# #     model = KerasClassifier(
# #         model=model,
# #         epochs=hparams.get("epochs", 10),
# #         batch_size=hparams.get("batch_size", 16),
# #         validation_split=hparams.get("validation_split", 0.2),
# #         # validation_data=hparams.get("validation_data"),
# #         # callbacks=[
# #         #     # ModelCheckpoint(),
# #         #     # EarlyStopping(
# #         #     #     monitor="val_binary_accuracy",  # "val_loss", "val_binary_accuracy"
# #         #     #     min_delta=0,
# #         #     #     patience=10,
# #         #     #     verbose=0,
# #         #     #     mode="auto",
# #         #     #     restore_best_weights=True,
# #         #     # ),
# #         # ],
# #         optimizer=model.optimizer,
# #         loss=model.loss,
# #         metrics=model.metrics,
# #         # use_multiprocessing=True,
# #         verbose=hparams.get("verbose", True),
# #     )
# #     return model


# # def keras_model_show(model, model_dir):
# #     """
# #     Visualising keras model architecture
# #     """
# #     # Making dst_dir (if necessary)
# #     os.makedirs(model_dir, exist_ok=True)
# #     # Making architecture image
# #     keras_plot(
# #         model,
# #         to_file=os.path.join(model_dir, "model.png"),
# #         show_shapes=True,
# #         show_dtype=False,
# #         show_layer_names=True,
# #         rankdir="TB",
# #         expand_nested=True,
# #         dpi=500,
# #         show_layer_activations=True,
# #         show_trainable=False,
# #     )


# # def keras_model_train(model, x_train, y_train, model_dir, hparams={}):
# #     """
# #     Training keras model
# #     """
# #     # Making dst_dir (if necessary)
# #     os.makedirs(model_dir, exist_ok=True)
# #     # Training keras model
# #     history = model.fit(
# #         x_train,
# #         y_train,
# #         epochs=hparams.get("epochs", 10),
# #         batch_size=hparams.get("batch_size", 16),
# #         validation_split=hparams.get("validation_split", 0.2),
# #         validation_data=hparams.get("validation_data"),
# #         # callbacks=[
# #         #     # ModelCheckpoint(),
# #         #     # EarlyStopping(
# #         #     #     monitor="val_binary_accuracy",  # "val_loss", "val_binary_accuracy"
# #         #     #     min_delta=0,
# #         #     #     patience=10,
# #         #     #     verbose=0,
# #         #     #     mode="auto",
# #         #     #     restore_best_weights=True,
# #         #     # ),
# #         # ],
# #         use_multiprocessing=hparams.get("use_multiprocessing", True),
# #         verbose=hparams.get("verbose", True),
# #     )
# #     # Saving model
# #     model.save(os.path.join(model_dir, "model.keras"))
# #     # Saving history
# #     joblib.dump(history, os.path.join(model_dir, "training_history.sav"))


# # def keras_model_predict(model_dir, X, y_true, pcutoff):
# #     # Loading keras model
# #     model = keras_load_model(os.path.join(model_dir, "model.keras"))
# #     # Getting probabilitites from model
# #     y_prob = model.predict(X).flatten()
# #     # Getting outcomes from probabilities using pcutoff
# #     y_pred = (y_prob > pcutoff).astype(int)
# #     # Returning eval_df df
# #     return pd.DataFrame(
# #         {
# #             BEHAV_ACTUAL_COL: y_true,
# #             BEHAV_PROB_COL: y_prob,
# #             BEHAV_PRED_COL: y_pred,
# #         }
# #     )


# #################################################
# #           SIMBA EVALUATE RESULTS
# #################################################


# def evalCSV(scored_fp, simba_fp, out_fp, behavs_ls, pcutoff, min_bout):
#     """
#     Merging the machine results with the actual scored data
#     """
#     name = getName(scored_fp)
#     # Reading the ml df and scored df csv files
#     scored_df = pd.read_csv(scored_fp, header=0, index_col=0)
#     simba_df = pd.read_csv(simba_fp, header=0, index_col=0)
#     # Sorting index (frames)
#     scored_df = scored_df.sort_index()
#     # Adding predicted labels to the scored df
#     for behav in behavs_ls:
#         # SETTING ACTUAL LABEL
#         if behav not in scored_df.columns:
#             print(f"SCORED {behav} not in {name}")
#             scored_df[behav] = 0
#         scored_df = scored_df.rename(columns={behav: f"{behav}_actual"})
#         # SETTING ML PREDICTED PROBABILITY
#         scored_df[f"{behav}_prob"] = simba_df[f"Probability_{behav}"]
#         # SETTING ML PREDICTED LABEL
#         # Setting label according to pcutoff
#         scored_df[f"{behav}_pred"] = (scored_df[f"{behav}_prob"] >= pcutoff).astype(int)
#         # Adjusting label according to min_bout
#         # Making ID for each predicted bout
#         scored_df["ID"] = (
#             scored_df[f"{behav}_pred"] != scored_df[f"{behav}_pred"].shift()
#         ).cumsum()
#         # Filtering for predicted non-behaviour and getting number for frames for each bout
#         x = (
#             scored_df[scored_df[f"{behav}_pred"] == 0]
#             .groupby("ID")
#             .agg({"Timestamp": "count"})
#         )
#         # Getting IDs of any bouts with less frames than "min_bout"
#         min_bout_ids = x[x["Timestamp"] < min_bout].index
#         # Setting bouts with these IDs as true behaviour
#         scored_df.loc[scored_df["ID"].isin(min_bout_ids), f"{behav}_pred"] = 1
#     # Dropping temporary ID column
#     scored_df = scored_df.drop(columns="ID")
#     # Saving the new scored df file (now has ML predicted outcomes)
#     scored_df.to_csv(out_fp)


# def evalPlot(eval_fp, out_fp, behavs_ls):
#     """
#     For each behaviour, plotting the actual label and the ML probability result on a subplot
#     """
#     name = IOMixin.get_name(eval_fp)
#     # Reading the eval csv
#     df = pd.read_csv(eval_fp, header=0, index_col=0)
#     # Initialising the plot
#     fig, axes = plt.subplots(nrows=len(behavs_ls), ncols=1, figsize=(8, 10))
#     for i, behav in enumerate(behavs_ls):
#         # Plotting actual behaviour classification
#         sns.lineplot(
#             data=df,
#             x="Timestamp",
#             y=f"{behav}_actual",
#             ax=axes[i],
#             color="red",
#             alpha=0.4,
#         )
#         # Plotting ML predicted behaviour probability
#         sns.lineplot(
#             data=df,
#             x="Timestamp",
#             y=f"{behav}_prob",
#             ax=axes[i],
#             color="blue",
#             alpha=0.4,
#         )
#     # Adding figure titles and labels
#     fig.suptitle("ML prob against (blue) acutal label (red)")
#     fig.savefig(out_fp)


# #################################################
# #           SIMBA EVAL AGGREGATION
# #################################################


# def combineEvalCSV(eval_dir):
#     """
#     Concatenating all frame-by-frame evaluation csv files into a single file named "TOTAL.csv".
#     """
#     # Making total frame-by-frame comparison df of ALL frames across ALL videos
#     silent_remove(os.path.join(eval_dir, "TOTAL.csv"))
#     df = pd.DataFrame()
#     for fp in os.listdir(eval_dir):
#         name = IOMixin.get_name(fp)
#         df = pd.concat(
#             (
#                 df,
#                 pd.read_csv(os.path.join(eval_dir, fp), index_col=0, header=[0]).assign(
#                     exp=name
#                 ),
#             ),
#             axis=0,
#         )
#     # Saving to TOTAL.csv file
#     df.to_csv(os.path.join(eval_dir, "TOTAL.csv"))
#     return df
#     return df
#     return df
