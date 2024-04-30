"""
_summary_
"""

import functools
import os
import re
from multiprocessing import Pool
from typing import Any, Callable

import numpy as np
import pandas as pd
import seaborn as sns
from behavysis_core.constants import (
    ANALYSIS_DIR,
    DIAGNOSTICS_DIR,
    FOLDERS,
    STR_DIV,
    TEMP_DIR,
)
from behavysis_core.data_models.experiment_configs import ConfigsAuto, ExperimentConfigs
from behavysis_core.mixins.df_io_mixin import DFIOMixin
from behavysis_core.mixins.io_mixin import IOMixin
from behavysis_core.mixins.multiproc_mixin import MultiprocMixin
from natsort import natsort_keygen, natsorted

from behavysis_pipeline.pipeline.experiment import Experiment
from behavysis_pipeline.processes.run_dlc import RunDLC


class Project:
    """
    A project is used to process and analyse many experiments at the same time.

    Expected filesystem hierarchy of project directory is below:
    ```
        - dir
            - 0_configs
                - exp1.json
                - exp2.json
                - ...
            - 1_raw_vid
                - .mp4
                - exp2.mp4
                - ...
            - 2_formatted_vid
                - exp1.mp4
                - exp2.mp4
                - ...
            - 3_dlc
                - exp1.feather
                - exp2.feather
                - ...
            - 4_preprocessed
                - exp1.feather
                - exp2.feather
                - ...
            - 5_features_extracted
                - exp1.feather
                - exp2.feather
                - ...
            - 6_predicted_behavs
                - exp1.feather
                - exp2.feather
                - ...
            - 7_scored_behavs
                - exp1.feather
                - exp2.feather
                - ...
            - diagnostics
                - <outputs for every tranformation>.csv
            - analysis
                - thigmotaxis
                    - fbf
                        - exp1.feather
                        - exp2.feather
                        - ...
                    - summary
                        - exp1.feather
                        - exp2.feather
                        - ...
                        - __ALL.feather
                    - binned_5
                        - exp1.feather
                        - exp2.feather
                        - ...
                        - __ALL.feather
                    - binned_5_plot
                        - exp1.feather
                        - exp2.feather
                        - ...
                        - __ALL.feather
                    - binned_30
                        - exp1.feather
                        - exp2.feather
                        - ...
                        - __ALL.feather
                    - binned_30_plot
                        - exp1.feather
                        - exp2.feather
                        - ...
                        - __ALL.feather
                    - binned_custom
                        - exp1.feather
                        - exp2.feather
                        - ...
                        - __ALL.feather
                    - binned_custom_plot
                        - exp1.feather
                        - exp2.feather
                        - ...
                        - __ALL.feather
                - speed
                    - fbf
                    - summary
                    - binned_5
                    - binned_5_plot
                    - ...
                - EPM
                - SFC
                - 3Chamber
                - Withdrawal
                - ...
            - evaluate
                - keypoints_plot
                    - exp1.feather
                    - exp2.feather
                    - ...
                - eval_vid
                    - exp1.feather
                    - exp2.feather
                    - ...
    ```

    Params:
        dir: The filepath of the project directory. Can be relative to current directory
        or absolute.

    Attributes:
        dir: The filepath of the project directory. Can be relative to current directory
        or absolute.
        experiments: The experiments that have been loaded into the project.

    Raises:
        ValueError: The givendir filepath does not exist.
    """

    root_dir: str
    experiments: dict[str, Experiment]
    nprocs: int

    def __init__(self, root_dir: str) -> None:
        """
        Make a Project instance.
        """
        # Assertion: project directory must exist
        if not os.path.isdir(root_dir):
            raise ValueError(
                f'Error: The folder, "{root_dir}" does not exist.\n'
                + "Please specify a folder that exists. Ensure you have the correct"
                + "forward-slashes or back-slashes for the path name."
            )
        self.root_dir = os.path.abspath(root_dir)
        self.experiments = {}
        self.nprocs = 4

    #####################################################################
    #               GETTER METHODS
    #####################################################################

    def get_experiment(self, name: str) -> Experiment:
        """
        Gets the experiment with the given name

        Parameters
        ----------
        name : str
            The experiment name.

        Returns
        -------
        Experiment
            The experiment.

        Raises
        ------
        ValueError
            Experiment with the given name does not exist.
        """
        if name in self.experiments:
            return self.experiments[name]
        raise ValueError(
            f'Experiment with the name "{name}" does not exist in the project.'
        )

    def get_experiments(self) -> list[Experiment]:
        """
        Gets the ordered (natsorted) list of Experiment instances in the Project.

        Returns
        -------
        list[Experiment]
            The list of all Experiment instances stored in the Project instance.
        """
        return [self.experiments[i] for i in natsorted(self.experiments)]

    #####################################################################
    #               PROJECT PROCESSING SCAFFOLD METHODS
    #####################################################################

    @staticmethod
    def _process_scaffold_mp_worker(args_tuple: tuple):
        method, exp, args, kwargs = args_tuple
        return method(exp, *args, **kwargs)

    def _process_scaffold_mp(
        self, method: Callable, *args: Any, **kwargs: Any
    ) -> list[dict]:
        """
        Processes an experiment with the given `Experiment` method and records
        the diagnostics of the process in a MULTI-PROCESSING way.

        Parameters
        ----------
        method : Callable
            The `Experiment` class method to run.

        Notes
        -----
        Can call any `Experiment` methods instance.
        Effectively, `method` gets called with:
        ```
        # exp is a Experiment instance
        method(exp, *args, **kwargs)
        ```
        """
        # Create a Pool of processes
        with Pool(processes=self.nprocs) as p:
            # Apply method to each experiment in self.get_experiments() in parallel
            return p.map(
                Project._process_scaffold_mp_worker,
                [(method, exp, args, kwargs) for exp in self.get_experiments()],
            )

    def _process_scaffold_sp(
        self, method: Callable, *args: Any, **kwargs: Any
    ) -> list[dict]:
        """
        Processes an experiment with the given `Experiment` method and records
        the diagnostics of the process in a SINGLE-PROCESSING way.

        Parameters
        ----------
        method : Callable
            The experiment `Experiment` class method to run.

        Notes
        -----
        Can call any `Experiment` instance method.
        Effectively, `method` gets called with:
        ```
        # exp is a Experiment instance
        method(exp, *args, **kwargs)
        ```
        """
        # Processing all experiments and storing process outcomes as list of dicts
        return [method(exp, *args, **kwargs) for exp in self.get_experiments()]

    def _process_scaffold(self, method: Callable, *args: Any, **kwargs: Any) -> None:
        """
        Runs the given method on all experiments in the project.
        """
        # Choosing whether to run the scaffold function in single or multi-processing mode
        if self.nprocs == 1:
            scaffold_func = self._process_scaffold_sp
        else:
            scaffold_func = self._process_scaffold_mp
        # Running the scaffold function
        # Starting
        print(f"Running {method.__name__}")
        # Running
        dd_ls = scaffold_func(method, *args, **kwargs)
        # Processing all experiments
        df = (
            pd.DataFrame(dd_ls).set_index("experiment").sort_index(key=natsort_keygen())
        )
        # Updating the diagnostics file at each step
        self.save_diagnostics(method.__name__, df)
        # Finishing
        print(f"Finished {method.__name__}!\n{STR_DIV}\n{STR_DIV}\n")

    #####################################################################
    #               BATCH PROCESSING METHODS
    #####################################################################

    @functools.wraps(Experiment.update_configs)
    def update_configs(self, *args, **kwargs) -> None:
        """
        Batch processing for corresponding [Experiment method](experiment.md#behavysis_pipeline.pipeline.Experiment.update_configs)
        """
        method = Experiment.update_configs
        self._process_scaffold(method, *args, **kwargs)

    @functools.wraps(Experiment.format_vid)
    def format_vid(self, *args, **kwargs) -> None:
        """
        Batch processing for corresponding [Experiment method](experiment.md#behavysis_pipeline.pipeline.Experiment.format_vid)
        """
        method = Experiment.format_vid
        self._process_scaffold(method, *args, **kwargs)

    @functools.wraps(Experiment.run_dlc)
    def run_dlc(self, gputouse: int = None, overwrite: bool = False) -> None:
        """
        Batch processing for corresponding [Experiment method](experiment.md#behavysis_pipeline.pipeline.Experiment.run_dlc)
        """
        nprocs = len(MultiprocMixin.get_gpu_ids()) if gputouse is None else 1
        # Getting the experiments to run DLC on
        exp_ls = self.get_experiments()
        # If overwrite is false, filtering for only experiments that need to be run
        if not overwrite:
            exp_ls = [exp for exp in exp_ls if not os.path.isfile(exp.get_fp("3_dlc"))]
        # Splitting the experiments into batches
        exp_batches_ls = np.array_split(exp_ls, nprocs)
        # Running DLC on each batch of experiments
        with Pool(processes=nprocs) as p:
            p.starmap(
                RunDLC.ma_dlc_analyse_batch,
                [
                    (
                        [exp.get_fp("2_formatted_vid") for exp in exp_batch],
                        os.path.join(self.root_dir, "3_dlc"),
                        os.path.join(self.root_dir, "0_configs"),
                        os.path.join(self.root_dir, TEMP_DIR),
                        gputouse,
                        overwrite,
                    )
                    for exp_batch in exp_batches_ls
                ],
            )

    @functools.wraps(Experiment.calculate_params)
    def calculate_params(self, *args, **kwargs) -> None:
        """
        Batch processing for corresponding [Experiment method](experiment.md#behavysis_pipeline.pipeline.Experiment.calculate_params)
        """
        method = Experiment.calculate_params
        self._process_scaffold(method, *args, **kwargs)

    @functools.wraps(Experiment.preprocess)
    def preprocess(self, *args, **kwargs) -> None:
        """
        Batch processing for corresponding [Experiment method](experiment.md#behavysis_pipeline.pipeline.Experiment.preprocess)
        """
        method = Experiment.preprocess
        self._process_scaffold(method, *args, **kwargs)

    @functools.wraps(Experiment.extract_features)
    def extract_features(self, *args, **kwargs) -> None:
        """
        Batch processing for corresponding [Experiment method](experiment.md#behavysis_pipeline.pipeline.Experiment.extract_features)
        """
        method = Experiment.extract_features
        self._process_scaffold(method, *args, **kwargs)

    @functools.wraps(Experiment.classify_behaviours)
    def classify_behaviours(self, *args, **kwargs) -> None:
        """
        Batch processing for corresponding [Experiment method](experiment.md#behavysis_pipeline.pipeline.Experiment.classify_behaviours)
        """
        # TODO: handle reading the model file whilst in multiprocessing. Current fix is single processing.
        nprocs = self.nprocs
        self.nprocs = 1
        method = Experiment.classify_behaviours
        self._process_scaffold(method, *args, **kwargs)
        self.nprocs = nprocs

    @functools.wraps(Experiment.export_behaviours)
    def export_behaviours(self, *args, **kwargs) -> None:
        """
        Batch processing for corresponding [Experiment method](experiment.md#behavysis_pipeline.pipeline.Experiment.export_behaviours)
        """
        method = Experiment.export_behaviours
        self._process_scaffold(method, *args, **kwargs)

    @functools.wraps(Experiment.export_feather)
    def export_feather(self, *args, **kwargs) -> None:
        """
        Batch processing for corresponding [Experiment method](experiment.md#behavysis_pipeline.pipeline.Experiment.export_feather)
        """
        method = Experiment.export_feather
        self._process_scaffold(method, *args, **kwargs)

    @functools.wraps(Experiment.evaluate)
    def evaluate(self, *args, **kwargs) -> None:
        """
        Batch processing for corresponding [Experiment method](experiment.md#behavysis_pipeline.pipeline.Experiment.evaluate)
        """
        method = Experiment.evaluate
        self._process_scaffold(method, *args, **kwargs)

    @functools.wraps(Experiment.analyse)
    def analyse(self, *args, **kwargs) -> None:
        """
        Batch processing for corresponding [Experiment method](experiment.md#behavysis_pipeline.pipeline.Experiment.analyse)
        """
        method = Experiment.analyse
        self._process_scaffold(method, *args, **kwargs)

    #####################################################################
    #               DIAGNOSTICS DICT METHODS
    #####################################################################

    def load_diagnostics(self, name: str) -> pd.DataFrame:
        """
        Reads the data from the diagnostics file with the given name.

        Parameters
        ----------
        name : str
            The name of the diagnostics file to overwrite and open.

        Returns
        -------
        pd.DataFrame
            The pandas DataFrame of the diagnostics file.
        """
        fp = os.path.join(self.root_dir, DIAGNOSTICS_DIR, f"{name}.csv")
        return pd.read_csv(fp, index_col=0)

    def save_diagnostics(self, name: str, df: pd.DataFrame) -> None:
        """
        Writes the given data to a diagnostics file with the given name.

        Parameters
        ----------
        name : str
            The name of the diagnostics file to overwrite and open.
        df : pd.DataFrame
            The pandas DataFrame to write to the diagnostics file.
        """
        # Getting diagnostics filepath for given name
        fp = os.path.join(self.root_dir, DIAGNOSTICS_DIR, f"{name}.csv")
        # Making a folder if it does not exist
        os.makedirs(os.path.split(fp)[0], exist_ok=True)
        # Writing diagnostics file
        df.to_csv(fp)

    #####################################################################
    #                CONFIGS DIAGONOSTICS METHODS
    #####################################################################

    def collate_configs_auto(self) -> None:
        """
        Collates the auto fields of the configs of all experiments into a DataFrame.
        """
        # Getting all the auto field keys
        auto_field_keys = ConfigsAuto.get_field_names(ConfigsAuto)
        # Making a DataFrame to store all the auto fields for each experiment
        df_configs = pd.DataFrame(
            index=[exp.name for exp in self.get_experiments()],
            # columns=ConfigsAuto.model_fields.keys(),
            columns=["_".join(i) for i in auto_field_keys],
        )
        # Collating all the auto fields for each experiment
        for exp in self.get_experiments():
            configs = ExperimentConfigs.read_json(exp.get_fp("0_configs"))
            for i in auto_field_keys:
                val = configs.auto
                for j in i:
                    val = getattr(val, j)
                df_configs.loc[exp.name, "_".join(i)] = val
        # Saving the collated auto fields DataFrame to diagnostics folder
        self.save_diagnostics("collated_configs_auto", df_configs)

        # Making and saving histogram plots of all the auto fields
        g = sns.FacetGrid(
            data=df_configs.melt(), col="variable", sharex=False, col_wrap=4
        )
        g.map(sns.histplot, "value", bins=20)
        g.set_titles("{col_name}")
        g.savefig(
            os.path.join(
                self.root_dir, DIAGNOSTICS_DIR, "collated_configs_auto_hist.png"
            )
        )
        g.figure.clf()

    #####################################################################
    #               IMPORT EXPERIMENTS METHODS
    #####################################################################

    def import_experiment(self, name: str) -> bool:
        """
        Adds an experiment with the given name to the .experiments dict.
        The key of this experiment in the `self.experiments` dict is "dir/name".
        If the experiment already exists in the project, it is not added.

        Parameters
        ----------
        name : str
            The experiment name.

        Returns
        -------
        bool
            Whether the experiment was imported or not.
            True if imported, False if not.
        """
        if not name in self.experiments:
            self.experiments[name] = Experiment(name, self.root_dir)
            return True
        return False

    def import_experiments(self) -> None:
        """
        Add all experiments in the project folder to the experiments dict.
        The key of each experiment in the .experiments dict is "name".
        Refer to Project.addExperiment() for details about how each experiment is added.
        """
        print(f"Searching project folder: {self.root_dir}\n")
        # Adding all experiments within given project dir
        failed = []
        for i in FOLDERS:
            folder = os.path.join(self.root_dir, i)
            # If folder does not exist, skip
            if not os.path.isdir(folder):
                continue
            # For each file in the folder
            for j in natsorted(os.listdir(folder)):
                if re.search(r"^\.", j):  # do not add hidden files
                    continue
                name = IOMixin.get_name(j)
                try:
                    self.import_experiment(name)
                except ValueError as e:  # do not add invalid files
                    print(f"failed: {i}    --    {j}:\n{e}")
                    failed.append(name)
        # Printing outcome of imported and failed experiments
        print("Experiments imported successfully:")
        print("\n".join([f"    - {i}" for i in self.experiments]), end="\n\n")
        print("Experiments failed to import:")
        print("\n".join([f"    - {i}" for i in failed]), end="\n\n")
        # Making diagnostics DataFrame of all the files associated with each experiment that exists
        # TODO: MAKE FASTER
        cols_ls = list(FOLDERS)
        rows_ls = list(self.experiments)
        shape = (len(rows_ls), len(cols_ls))
        dd_arr = np.apply_along_axis(
            lambda i: os.path.isfile(self.experiments[i[1]].get_fp(i[0])),
            axis=0,
            arr=np.array(np.meshgrid(cols_ls, rows_ls)).reshape((2, np.prod(shape))),
        ).reshape(shape)
        # Creating the diagnostics DataFrame
        dd_df = pd.DataFrame(dd_arr, index=rows_ls, columns=cols_ls)
        # Saving the diagnostics DataFrame
        self.save_diagnostics("import_experiments", dd_df)

    #####################################################################
    #            COMBINING ANALYSIS DATA ACROSS EXPS METHODS
    #####################################################################

    def collate_analysis_binned(self) -> None:
        """
        Combines an analysis of all the experiments together to generate combined h5 files for:
        - Each binned data. The index is (bin) and columns are (expName, indiv, measure).
        """
        # Initialising the process and printing the description
        description = "Combining binned analysis"
        print(f"{description}...")
        # dd_df = pd.DataFrame()

        # AGGREGATING BINNED DATA
        # NOTE: need a more robust way of getting the list of bin sizes
        analysis_dir = os.path.join(self.root_dir, ANALYSIS_DIR)
        configs = ExperimentConfigs.read_json(
            self.get_experiments()[0].get_fp("0_configs")
        )
        bin_sizes_sec = configs.user.analyse.bins_sec
        bin_sizes_sec = np.append(bin_sizes_sec, "custom")
        # Searching through all the analysis sub-folders
        for i in os.listdir(analysis_dir):
            if i == "aggregate_analysis":
                continue
            analysis_subdir = os.path.join(analysis_dir, i)
            for bin_i in bin_sizes_sec:
                total_df = pd.DataFrame()
                out_fp = os.path.join(analysis_subdir, f"__ALL_binned_{bin_i}.feather")
                for exp in self.get_experiments():
                    in_fp = os.path.join(
                        analysis_subdir, f"binned_{bin_i}", f"{exp.name}.feather"
                    )
                    if os.path.isfile(in_fp):
                        # Reading exp summary df
                        df = DFIOMixin.read_feather(in_fp)
                        # Prepending experiment name to column MultiIndex
                        df = pd.concat(
                            [df], keys=[exp.name], names=["experiment"], axis=1
                        )
                        # Concatenating total_df with df across columns
                        total_df = pd.concat([total_df, df], axis=1)
                    DFIOMixin.write_feather(total_df, out_fp)

    def collate_analysis_summary(self) -> None:
        """
        Combines an analysis of all the experiments together to generate combined h5 files for:
        - The summary data. The index is (expName, indiv, measure) and columns are
        (statistics -e.g., mean).
        """
        # Initialising the process and printing the description
        description = "Combining summary analysis"
        print(f"{description}...")
        # dd_df = pd.DataFrame()

        # AGGREGATING SUMMARY DATA
        analysis_dir = os.path.join(self.root_dir, ANALYSIS_DIR)
        # Searching through all the analysis sub-folders
        for i in os.listdir(analysis_dir):
            if i == "aggregate_analysis":
                continue
            analysis_subdir = os.path.join(analysis_dir, i)
            total_df = pd.DataFrame()
            out_fp = os.path.join(analysis_subdir, "__ALL_summary.feather")
            for exp in self.get_experiments():
                in_fp = os.path.join(analysis_subdir, "summary", f"{exp.name}.feather")
                if os.path.isfile(in_fp):
                    # Reading exp summary df
                    df = DFIOMixin.read_feather(in_fp)
                    # Prepending experiment name to index MultiIndex
                    df = pd.concat([df], keys=[exp.name], names=["experiment"], axis=0)
                    # Concatenating total_df with df down rows
                    total_df = pd.concat([total_df, df], axis=0)
            DFIOMixin.write_feather(total_df, out_fp)
            DFIOMixin.write_feather(total_df, out_fp)
