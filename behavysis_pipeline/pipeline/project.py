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
from behavysis_core.mixins.df_io_mixin import DFIOMixin
from behavysis_core.mixins.io_mixin import IOMixin
from behavysis_core.mixins.multiproc_mixin import MultiprocMixin
from behavysis_core.utils.constants import (
    ANALYSIS_DIR,
    DIAGNOSTICS_DIR,
    FOLDERS,
    STR_DIV,
    TEMP_DIR,
)
from natsort import natsort_keygen, natsorted

from behavysis_pipeline.pipeline.experiment import BehavysisExperiment
from behavysis_pipeline.pipeline.experiment_configs import ExperimentConfigs
from behavysis_pipeline.processes.run_dlc import RunDLC


class BehavysisProject:
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

    def __init__(self, root_dir: str) -> None:
        """
        Make a BehavysisProject instance.
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

        # Making batch processing methods dynamically for BehavysisExperiment methods
        self.process_scaffold = self.process_scaffold_mp
        method_names_ls = [
            i for i in dir(BehavysisExperiment) if not re.search(r"^_", i)
        ]
        for method_name in method_names_ls:
            self._batch_process_factory(method_name)
        # Setting custom batch processing methods
        self.run_dlc = self._run_dlc

    def _batch_process_factory(self, method_name: str):
        method = getattr(BehavysisExperiment, method_name)

        @functools.wraps(method)
        def batch_process(*args, **kwargs):
            # Starting
            print(f"Running {method_name}")
            # Running
            self.process_scaffold(method, *args, **kwargs)
            # Finishing
            print(f"Finished {method_name}!")
            print(f"{STR_DIV}\n{STR_DIV}\n")

        setattr(self, method_name, batch_process)

    #####################################################################
    #               GETTER METHODS
    #####################################################################

    def get_experiment(self, name: str) -> BehavysisExperiment:
        """
        Gets the experiment with the given name

        Parameters
        ----------
        name : str
            The experiment name.

        Returns
        -------
        BehavysisExperiment
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

    def get_experiments(self) -> list[BehavysisExperiment]:
        """
        Gets the ordered (natsorted) list of Experiment instances in the BehavysisProject.

        Returns
        -------
        list[BehavysisExperiment]
            The list of all BehavysisExperiment instances stored in the BehavysisProject instance.
        """
        return [self.experiments[i] for i in natsorted(self.experiments)]

    #####################################################################
    #               PROJECT PROCESSING SCAFFOLD METHODS
    #####################################################################

    @staticmethod
    def _process_scaffold_worker(args_tuple: tuple):
        method, exp, args, kwargs = args_tuple
        return method(exp, *args, **kwargs)

    def process_scaffold_mp(
        self,
        method: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Processes an experiment with the given `BehavysisExperiment` method and records the diagnostics
        of the process in a MULTI-PROCESSING way.

        Parameters
        ----------
        method : Callable
            The `BehavysisExperiment` method to run.

        Notes
        -----
        Can call any `BehavysisExperiment` methods instance.
        Effectively, `method` gets called with:
        ```
        # exp is a BehavysisExperiment instance
        method(exp, *args, **kwargs)
        ```
        """
        # Initialising diagnostics dataframe
        df = pd.DataFrame()

        # TODO: maybe have a try-except-finally block to ensure that the diagnostics file is saved
        # Create a Pool of processes
        with Pool(processes=self.nprocs) as p:
            # Apply method to each experiment in self.get_experiments() in parallel
            results = p.map(
                BehavysisProject._process_scaffold_worker,
                [(method, exp, args, kwargs) for exp in self.get_experiments()],
            )

        # Processing all experiments
        for dd in results:
            # Converting outcomes dict to dataframe
            dd = pd.DataFrame(pd.Series(dd)).transpose().set_index("experiment")
            # Adding outcomes dict to diagnostics dataframe
            df = pd.concat([df, dd], axis=0).sort_index(key=natsort_keygen())
            # Updating the diagnostics file at each step
            self.save_diagnostics(method.__name__, df)

    def process_scaffold_sp(
        self,
        method: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Processes an experiment with the given `BehavysisExperiment` method and records the diagnostics
        of the process in a SINGLE-PROCESSING way.

        Parameters
        ----------
        method : Callable
            The experiment `BehavysisExperiment` method to run.

        Notes
        -----
        Can call any `BehavysisExperiment` instance method.
        Effectively, `method` gets called with:
        ```
        # exp is a BehavysisExperiment instance
        method(exp, *args, **kwargs)
        ```
        """
        # Initialising diagnostics dataframe
        df = pd.DataFrame()
        # Processing all experiments
        for exp in self.get_experiments():
            # Processing and storing process outcomes as dict
            dd = method(exp, *args, **kwargs)
            # Converting outcomes dict to dataframe
            dd = pd.DataFrame(pd.Series(dd)).transpose().set_index("experiment")
            # Adding outcomes dict to diagnostics dataframe
            df = pd.concat([df, dd], axis=0).sort_index(key=natsort_keygen())
            # Updating the diagnostics file at each step
            self.save_diagnostics(method.__name__, df)

    #####################################################################
    #               CUSTOM BATCH DLC PROCESSING METHOD
    #####################################################################

    def _run_dlc(self, gputouse: int = None, overwrite: bool = False) -> None:
        """
        Custom batch processing function for DLC.
        """
        nprocs = len(MultiprocMixin.get_gpu_ids())
        # nprocs = 1
        # Getting the experiments to run DLC on
        exp_ls = self.get_experiments()
        # If overwrite is false, filtering for only experiments that need to be run
        if not overwrite:
            exp_ls = [exp for exp in exp_ls if not exp.check_fp("3_dlc")]
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

        # RunDLC.ma_dlc_analyse_batch(
        #     in_fp_ls=[exp.get_fp("2_formatted_vid") for exp in self.get_experiments()],
        #     out_dir=os.path.join(self.root_dir, "3_dlc"),
        #     configs_dir=os.path.join(self.root_dir, "0_configs"),
        #     temp_dir=os.path.join(self.root_dir, TEMP_DIR),
        #     gputouse=gputouse,
        #     overwrite=overwrite,
        # )

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
            self.experiments[name] = BehavysisExperiment(name, self.root_dir)
            return True
        return False

    def import_experiments(self) -> None:
        """
        Add all experiments in the project folder to the experiments dict.
        The key of each experiment in the .experiments dict is "name".
        Refer to BehavysisProject.addExperiment() for details about how each experiment is added.
        """
        print(f"Searching project folder: {self.root_dir}\n")
        # Adding all experiments within given project dir
        imported = []
        failed = []
        for i in FOLDERS:
            dir_folder = os.path.join(self.root_dir, i)
            if os.path.isdir(dir_folder):
                for j in natsorted(os.listdir(dir_folder)):
                    if j[0] == ".":  # do not add hidden files
                        continue
                    name = IOMixin.get_name(j)
                    try:
                        if self.import_experiment(name):
                            imported.append(name)
                    except ValueError as e:  # do not add invalid files
                        print(f"failed: {i}    --    {j}")
                        print(e)
                        failed.append(name)
        # Printing outcome of imported and failed experiments
        print("Experiments imported successfully:")
        for i in imported:
            print(f"    - {i}")
        print("")
        print("Experiments failed to import:")
        for i in failed:
            print(f"    - {i}")
        print("")
        # Making diagnostics DataFrame of all the files associated with each experiment that exists
        dd_df = pd.DataFrame(
            index=list(self.experiments.keys()), columns=list(FOLDERS.keys())
        )
        # TODO: MAKE FASTER
        # for exp in self.get_experiments():
        #     for folder in FOLDERS:
        #         dd_df.loc[exp.name, folder] = exp.check_fp(folder)
        self.save_diagnostics("importExperiments.csv", dd_df)

    #####################################################################
    #            COMBINING ANALYSIS DATA ACROSS EXPS METHODS
    #####################################################################

    def combine_analysis_binned(self) -> None:
        """
        Combines an analysis of all the experiments together to generate combined h5 files for:
        - Each binned data. The index is (bin) and columns are (expName, indiv, measure).

        Parameters
        ----------
        overwrite : bool
            Whether to overwrite the output file (if it exists).
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

    def combine_analysis_summary(self) -> None:
        """
        Combines an analysis of all the experiments together to generate combined h5 files for:
        - The summary data. The index is (expName, indiv, measure) and columns are
        (statistics -e.g., mean).

        Parameters
        ----------
        overwrite : bool
            Whether to overwrite the output file (if it exists).
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
