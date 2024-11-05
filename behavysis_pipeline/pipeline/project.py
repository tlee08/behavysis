"""
_summary_
"""

import functools
import logging
import os
import re
from multiprocessing import Pool
from typing import Any, Callable

import dask
import numpy as np
import pandas as pd
import seaborn as sns
from behavysis_core.constants import (
    ANALYSE_DIR,
    DIAGNOSTICS_DIR,
    STR_DIV,
    TEMP_DIR,
    Folders,
)
from behavysis_core.df_classes.df_mixin import DFMixin
from behavysis_core.mixins.diagnostics_mixin import DiagnosticsMixin
from behavysis_core.mixins.io_mixin import IOMixin
from behavysis_core.mixins.multiproc_mixin import MultiprocMixin
from behavysis_core.pydantic_models.experiment_configs import (
    ConfigsAuto,
    ExperimentConfigs,
)
from dask.distributed import LocalCluster
from natsort import natsort_keygen, natsorted

from behavysis_pipeline.pipeline.experiment import Experiment
from behavysis_pipeline.processes.run_dlc import RunDLC
from behavysis_pipeline.utils.dask_utils import cluster_proc_contxt


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

    Attributes
    ----------
        root_dir : str
            The filepath of the project directory. Can be relative to
            current dir or absolute dir.
        experiments : dict[str, Experiment]
            The experiments that have been loaded into the project.
        nprocs : int
            The number of processes to use for multiprocessing.
    """

    root_dir: str
    experiments: dict[str, Experiment]
    nprocs: int

    def __init__(self, root_dir: str) -> None:
        """
        Make a Project instance.

        Parameters
        ----------
        root_dir : str
            The filepath of the project directory. Can be relative to
            current dir or absolute dir.
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
    # GETTER METHODS
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

    # @staticmethod
    # def _process_scaffold_mp_worker(args_tuple: tuple):
    #     method, exp, args, kwargs = args_tuple
    #     return method(exp, *args, **kwargs)

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
        # # Create a Pool of processes
        # with Pool(processes=self.nprocs) as p:
        #     # Apply method to each experiment in self.get_experiments() in parallel
        #     return p.map(
        #         Project._process_scaffold_mp_worker,
        #         [(method, exp, args, kwargs) for exp in self.get_experiments()],
        #     )
        # Starting a dask cluster
        with cluster_proc_contxt(
            LocalCluster(n_workers=self.nprocs, threads_per_worker=1)
        ):
            # Preparing all experiments for execution
            f_d_ls = [
                dask.delayed(method)(exp, *args, **kwargs)
                for exp in self.get_experiments()
            ]
            # Executing in parallel
            dd_ls = dask.compute(*f_d_ls)
        return dd_ls

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
        logging.info("Running %s", method.__name__)
        # Running
        dd_ls = scaffold_func(method, *args, **kwargs)
        if len(dd_ls) > 0:
            # Processing all experiments
            df = (
                pd.DataFrame(dd_ls)
                .set_index("experiment")
                .sort_index(key=natsort_keygen())
            )
            # Updating the diagnostics file at each step
            self._save_diagnostics(method.__name__, df)
            # Finishing
            logging.info("Finished %s!\n%s\n%s\n", method.__name__, STR_DIV, STR_DIV)

    #####################################################################
    #               BATCH PROCESSING METHODS
    #####################################################################

    @functools.wraps(Experiment.run_dlc)
    def run_dlc(self, gputouse: int | None = None, overwrite: bool = False) -> None:
        """
        Batch processing corresponding to
        [behavysis_pipeline.pipeline.experiment.Experiment.run_dlc][]

        Uses a multiprocessing pool to run DLC on each batch of experiments with each GPU
        natively as batch in the same spawned subprocess (a DLC subprocess is spawned).
        This is a slight tweak from the regular method of running
        each experiment separately with multiprocessing.
        """
        # If gputouse is not specified, using all GPUs
        if gputouse is None:
            gputouse_ls = MultiprocMixin.get_gpu_ids()
        else:
            gputouse_ls = [gputouse]
        nprocs = len(gputouse_ls)
        # Getting the experiments to run DLC on
        exp_ls = self.get_experiments()
        # If overwrite is False, filtering for only experiments that need processing
        if not overwrite:
            exp_ls = [
                exp
                for exp in exp_ls
                if not os.path.isfile(exp.get_fp(Folders.DLC.value))
            ]

        # Running DLC on each batch of experiments with each GPU (given allocated GPU ID)
        # TODO: have error handling
        exp_batches_ls = np.array_split(np.array(exp_ls), nprocs)
        with Pool(processes=nprocs) as p:
            p.starmap(
                RunDLC.ma_dlc_analyse_batch,
                [
                    (
                        [exp.get_fp(Folders.FORMATTED_VID.value) for exp in exp_batch],
                        os.path.join(self.root_dir, Folders.DLC.value),
                        os.path.join(self.root_dir, Folders.CONFIGS.value),
                        os.path.join(self.root_dir, TEMP_DIR),
                        gputouse,
                        overwrite,
                    )
                    for gputouse, exp_batch in zip(gputouse_ls, exp_batches_ls)
                ],
            )

    # @functools.wraps(Experiment.classify_behaviours)
    # def classify_behaviours(self, *args, **kwargs) -> None:
    #     """
    #     Batch processing corresponding to
    #     [behavysis_pipeline.pipeline.experiment.Experiment.classify_behaviours][]
    #     """
    #     # TODO: handle reading the model file whilst in multiprocessing.
    #     # Current fix is single processing.
    #     # nprocs = self.nprocs
    #     # self.nprocs = 1
    #     method = Experiment.classify_behaviours
    #     self._process_scaffold(method, *args, **kwargs)
    #     # self.nprocs = nprocs

    #####################################################################
    # DIAGNOSTICS LOAD/SAVE METHODS
    #####################################################################

    def _load_diagnostics(self, name: str) -> pd.DataFrame:
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
        # Getting filepath
        fp = os.path.join(self.root_dir, DIAGNOSTICS_DIR, f"{name}.csv")
        # Reading from file
        return DiagnosticsMixin.load_diagnostics(fp)

    def _save_diagnostics(self, name: str, df: pd.DataFrame) -> None:
        """
        Writes the given data to a diagnostics file with the given name.

        Parameters
        ----------
        name : str
            The name of the diagnostics file to overwrite and open.
        df : pd.DataFrame
            The pandas DataFrame to write to the diagnostics file.
        """
        # Getting filepath
        fp = os.path.join(self.root_dir, DIAGNOSTICS_DIR, f"{name}.csv")
        # Writing to file
        DiagnosticsMixin.save_diagnostics(df, fp)

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
        if name not in self.experiments:
            self.experiments[name] = Experiment(name, self.root_dir)
            return True
        return False

    def import_experiments(self) -> None:
        """
        Add all experiments in the project folder to the experiments dict.
        The key of each experiment in the .experiments dict is "name".
        Refer to Project.addExperiment() for details about how each experiment is added.
        """
        logging.info("Searching project folder: %s\n", self.root_dir)
        # Adding all experiments within given project dir
        failed = []
        for f in Folders:
            folder = os.path.join(self.root_dir, f.value)
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
                    logging.info("failed: %s    --    %s:\n%s", f.value, j, e)
                    failed.append(name)
        # Logging outcome of imported and failed experiments
        logging.info("Experiments imported successfully:")
        logging.info("%s\n\n", "\n".join([f"    - {i}" for i in self.experiments]))
        logging.info("Experiments failed to import:")
        logging.info("%s\n\n", "\n".join([f"    - {i}" for i in failed]))
        # If there are no experiments, then return
        if not self.experiments:
            return
        # # Making diagnostics DataFrame of all the files associated with each experiment that exists
        # cols_ls = [f.value for f in Folders]
        # rows_ls = list(self.experiments)
        # shape = (len(rows_ls), len(cols_ls))
        # dd_arr = np.apply_along_axis(
        #     lambda i: os.path.isfile(self.experiments[i[1]].get_fp(i[0])),
        #     axis=0,
        #     arr=np.array(np.meshgrid(cols_ls, rows_ls)).reshape((2, np.prod(shape))),
        # ).reshape(shape)
        # # Creating the diagnostics DataFrame
        # dd_df = pd.DataFrame(dd_arr, index=rows_ls, columns=cols_ls)
        # # Saving the diagnostics DataFrame
        # self.save_diagnostics("import_experiments", dd_df)

    #####################################################################
    #                CONFIGS DIAGONOSTICS METHODS
    #####################################################################

    def collate_configs_auto(self) -> None:
        """
        Collates the auto fields of the configs of all experiments into a DataFrame.
        """
        # TODO: include diagnostics_df recording
        # Initialising the process and logging description
        description = "Combining binned analysis"
        logging.info("%s...", description)
        # Getting all the auto field keys
        auto_field_keys = ConfigsAuto.get_field_names(ConfigsAuto)
        # Making a DataFrame to store all the auto fields for each experiment
        df_configs = pd.DataFrame(
            index=[exp.name for exp in self.get_experiments()],
            columns=["_".join(i) for i in auto_field_keys],
        )
        # Collating all the auto fields for each experiment
        dd_ls = []
        for exp in self.get_experiments():
            try:
                configs = ExperimentConfigs.read_json(exp.get_fp(Folders.CONFIGS.value))
                for i in auto_field_keys:
                    val = configs.auto
                    for j in i:
                        val = getattr(val, j)
                    df_configs.loc[exp.name, "_".join(i)] = val  # type: ignore
                dd_ls.append({"experiment": exp.name, "outcome": "combined"})
            except Exception:
                dd_ls.append({"experiment": exp.name, "outcome": "ERROR: e"})
        # Imputing na values with -1
        df_configs = df_configs.fillna(-1)
        # Saving the collated auto fields DataFrame to diagnostics folder
        self._save_diagnostics("collated_configs_auto", df_configs)

        # Making and saving histogram plots of the numerical auto fields
        # NOTE: NOT including string frequencies, only numerical
        df_configs = df_configs.loc[:, df_configs.apply(pd.api.types.is_numeric_dtype)]
        g = sns.FacetGrid(
            data=df_configs.fillna(-1).melt(), col="variable", sharex=False, col_wrap=4
        )
        g.map(sns.histplot, "value", bins=10)
        g.set_titles("{col_name}")
        g.savefig(
            os.path.join(
                self.root_dir, DIAGNOSTICS_DIR, "collated_configs_auto_hist.png"
            )
        )
        g.figure.clf()

    #####################################################################
    #            COMBINING ANALYSIS DATA ACROSS EXPS METHODS
    #####################################################################

    def analyse_collate(self) -> None:
        """
        Combines an analysis of all the experiments together to generate combined files for:
        - Each binned data. The index is (bin) and columns are (expName, indiv, measure).
        - The summary data. The index is (expName, indiv, measure) and columns are
        (statistics -e.g., mean).
        """
        self._analyse_collate_binned()
        self._analyse_collate_summary()

    def _analyse_collate_binned(self) -> None:
        """
        Combines an analysis of all the experiments together to generate combined h5 files for:
        - Each binned data. The index is (bin) and columns are (expName, indiv, measure).
        """
        # Initialising the process and logging description
        description = "Combining binned analysis"
        logging.info("%s...", description)
        # AGGREGATING BINNED DATA
        # NOTE: need a more robust way of getting the list of bin sizes
        proj_analyse_dir = os.path.join(self.root_dir, ANALYSE_DIR)
        configs = ExperimentConfigs.read_json(
            self.get_experiments()[0].get_fp(Folders.CONFIGS.value)
        )
        bin_sizes_sec = configs.get_ref(configs.user.analyse.bins_sec)
        bin_sizes_sec = np.append(bin_sizes_sec, "custom")
        # Searching through all the analysis subdir
        for i in os.listdir(proj_analyse_dir):
            for bin_i in bin_sizes_sec:
                df_ls = []
                names_ls = []
                for exp in self.get_experiments():
                    in_fp = os.path.join(
                        proj_analyse_dir, i, f"binned_{bin_i}", f"{exp.name}.feather"
                    )
                    if os.path.isfile(in_fp):
                        df_ls.append(DFMixin.read_feather(in_fp))
                        names_ls.append(exp.name)
                # Concatenating total_df with df across columns, with experiment name to column MultiIndex
                if len(df_ls) > 0:
                    df = pd.concat(df_ls, keys=names_ls, names=["experiment"], axis=1)
                    out_fp = os.path.join(
                        proj_analyse_dir, i, f"__ALL_binned_{bin_i}.feather"
                    )
                    DFMixin.write_feather(df, out_fp)

    def _analyse_collate_summary(self) -> None:
        """
        Combines an analysis of all the experiments together to generate combined h5 files for:
        - The summary data. The index is (expName, indiv, measure) and columns are
        (statistics -e.g., mean).
        """
        # Initialising the process and logging description
        description = "Combining summary analysis"
        logging.info("%s...", description)
        # AGGREGATING SUMMARY DATA
        proj_analyse_dir = os.path.join(self.root_dir, ANALYSE_DIR)
        # Searching through all the analysis subdir
        for i in os.listdir(proj_analyse_dir):
            df_ls = []
            names_ls = []
            for exp in self.get_experiments():
                in_fp = os.path.join(
                    proj_analyse_dir, i, "summary", f"{exp.name}.feather"
                )
                if os.path.isfile(in_fp):
                    # Reading exp summary df
                    df_ls.append(DFMixin.read_feather(in_fp))
                    names_ls.append(exp.name)
            out_fp = os.path.join(proj_analyse_dir, i, "__ALL_summary.feather")
            # Concatenating total_df with df across columns, with experiment name to column MultiIndex
            if len(df_ls) > 0:
                total_df = pd.concat(df_ls, keys=names_ls, names=["experiment"], axis=0)
                DFMixin.write_feather(total_df, out_fp)


###################################################################################################
# DYNAMICALLY MAKING PROJECT PIPELINE METHODS (FROM EXPERIMENT METHODS) FOR PROJECT CLASS
###################################################################################################


def create_wrapped_method(func: Callable):
    @functools.wraps(func)
    def wrapper(self: Project, *args, **kwargs):
        self._process_scaffold(func, *args, **kwargs)

    return wrapper


# Dynamically add methods to Project class
for func in [
    Experiment.update_configs,
    Experiment.format_vid,
    Experiment.run_dlc,
    Experiment.calculate_parameters,
    Experiment.preprocess,
    Experiment.extract_features,
    Experiment.classify_behaviours,
    Experiment.export_behaviours,
    Experiment.export_feather,
    Experiment.analyse,
    Experiment.analyse_behavs,
    Experiment.analyse_combine,
    Experiment.evaluate,
]:
    func_name = func.__name__
    if func_name not in dir(Project):
        setattr(Project, func_name, create_wrapped_method(func))
