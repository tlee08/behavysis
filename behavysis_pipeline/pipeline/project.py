"""
_summary_
"""

import functools
import os
import re
from typing import Any, Callable

import dask
import numpy as np
import pandas as pd
import seaborn as sns
from dask.distributed import LocalCluster
from natsort import natsorted

from behavysis_pipeline.constants import (
    ANALYSIS_DIR,
    DIAGNOSTICS_DIR,
    STR_DIV,
    Folders,
)
from behavysis_pipeline.df_classes.analyse_agg_df import AnalyseBinnedDf, AnalyseSummaryDf
from behavysis_pipeline.df_classes.diagnostics_df import DiagnosticsDf
from behavysis_pipeline.pipeline.experiment import Experiment
from behavysis_pipeline.processes.run_dlc import RunDLC
from behavysis_pipeline.pydantic_models.configs import (
    ExperimentConfigs,
)
from behavysis_pipeline.utils.dask_utils import cluster_proc_contxt
from behavysis_pipeline.utils.io_utils import get_name
from behavysis_pipeline.utils.logging_utils import init_logger
from behavysis_pipeline.utils.multiproc_utils import get_gpu_ids


class Project:
    """
    A project is used to process and analyse many experiments at the same time.

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

    logger = init_logger(__name__)

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
                "Please specify a folder that exists. Ensure you have the correct"
                "forward-slashes or back-slashes for the path name."
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
        raise ValueError(f'Experiment with the name "{name}" does not exist in the project.')

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

    def _proc_scaff_mp(self, method: Callable, *args: Any, **kwargs: Any) -> list[dict]:
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
        exp is a Experiment instance
        method(exp, *args, **kwargs)
        ```
        """
        # Starting a dask cluster
        with cluster_proc_contxt(LocalCluster(n_workers=self.nprocs, threads_per_worker=1)):
            # Preparing all experiments for execution
            f_d_ls = [dask.delayed(method)(exp, *args, **kwargs) for exp in self.get_experiments()]
            # Executing in parallel
            dd_ls = list(dask.compute(*f_d_ls))
        return dd_ls

    def _proc_scaff_sp(self, method: Callable, *args: Any, **kwargs: Any) -> list[dict]:
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
        exp is a Experiment instance
        method(exp, *args, **kwargs)
        ```
        """
        # Processing all experiments and storing process outcomes as list of dicts
        return [method(exp, *args, **kwargs) for exp in self.get_experiments()]

    def _proc_scaff(self, method: Callable, *args: Any, **kwargs: Any) -> None:
        """
        Runs the given method on all experiments in the project.
        """
        # Choosing whether to run the scaffold function in single or multi-processing mode
        if self.nprocs == 1:
            scaffold_func = self._proc_scaff_sp
        else:
            scaffold_func = self._proc_scaff_mp
        # Running the scaffold function
        # Starting
        self.logger.info("Running %s", method.__name__)
        # Running
        dd_ls = scaffold_func(method, *args, **kwargs)
        if len(dd_ls) > 0:
            # Processing all experiments
            df = DiagnosticsDf.init_from_dd_ls(dd_ls)
            # Updating the diagnostics file at each step
            DiagnosticsDf.write(df, os.path.join(self.root_dir, DIAGNOSTICS_DIR, f"{method.__name__}.csv"))
            # Finishing
            self.logger.info("Finished %s!\n%s\n%s\n", method.__name__, STR_DIV, STR_DIV)

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
        self.logger.info("Searching project folder: %s\n", self.root_dir)
        # Storing file existences in {folder1: [file1, file2, ...], ...} format
        dd_dict = {}
        # Adding all experiments within given project dir
        for f in Folders:
            folder = os.path.join(self.root_dir, f.value)
            dd_dict[f.value] = []
            # If folder does not exist, skip
            if not os.path.isdir(folder):
                continue
            # For each file in the folder
            for j in natsorted(os.listdir(folder)):
                if re.search(r"^\.", j):  # do not add hidden files
                    continue
                name = get_name(j)
                try:
                    self.import_experiment(name)
                    dd_dict[f.value].append(name)
                except ValueError as e:  # do not add invalid files
                    self.logger.info("failed: %s    --    %s:\n%s", f.value, j, e)
        # Logging outcome of imported and failed experiments
        self.logger.info("Experiments imported successfully:")
        self.logger.info("\n%s\n", "\n".join([f"    - {i}" for i in self.experiments]))
        # Constructing dd_df from dd_dict
        dd_df = DiagnosticsDf.init_df(pd.Series(np.unique(np.concatenate(list(dd_dict.values())))))
        # Setting each (experiment, folder) pair to True if the file exists
        for folder in dd_dict:
            dd_df[folder] = False
            for exp_name in dd_dict[folder]:
                dd_df.loc[exp_name, folder] = True
        # Saving the diagnostics DataFrame
        DiagnosticsDf.write(dd_df, os.path.join(self.root_dir, DIAGNOSTICS_DIR, "import_experiments.csv"))

    #####################################################################
    #         BATCH PROCESSING WRAPPING EXPERIMENT METHODS
    #####################################################################

    @functools.wraps(Experiment.update_configs)
    def update_configs(self, *args, **kwargs):
        self._proc_scaff(Experiment.update_configs, *args, **kwargs)

    @functools.wraps(Experiment.format_vid)
    def format_vid(self, *args, **kwargs):
        self._proc_scaff(Experiment.format_vid, *args, **kwargs)

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
        # TODO: implement diagnostics
        # TODO: implement error handling
        # If gputouse is not specified, using all GPUs
        gputouse_ls = get_gpu_ids() if gputouse is None else [gputouse]
        nprocs = len(gputouse_ls)
        # Getting the experiments to run DLC on
        exp_ls = self.get_experiments()
        # If overwrite is False, filtering for only experiments that need processing
        if not overwrite:
            exp_ls = [exp for exp in exp_ls if not os.path.isfile(exp.get_fp(Folders.DLC.value))]
        # Running DLC on each batch of experiments with each GPU (given allocated GPU ID)
        exp_batches_ls = np.array_split(np.array(exp_ls), nprocs)
        # Starting a dask cluster
        with cluster_proc_contxt(LocalCluster(n_workers=nprocs, threads_per_worker=1)):
            # Preparing all experiments for execution
            f_d_ls = [
                dask.delayed(RunDLC.ma_dlc_analyse_batch)(
                    [exp.get_fp(Folders.FORMATTED_VID.value) for exp in exp_batch],
                    os.path.join(self.root_dir, Folders.DLC.value),
                    os.path.join(self.root_dir, Folders.CONFIGS.value),
                    gputouse,
                    overwrite,
                )
                for gputouse, exp_batch in zip(gputouse_ls, exp_batches_ls)
            ]
            # Executing in parallel
            list(dask.compute(*f_d_ls))

    @functools.wraps(Experiment.calculate_parameters)
    def calculate_parameters(self, *args, **kwargs):
        self._proc_scaff(Experiment.calculate_parameters, *args, **kwargs)

    @functools.wraps(Experiment.preprocess)
    def preprocess(self, *args, **kwargs):
        self._proc_scaff(Experiment.preprocess, *args, **kwargs)

    @functools.wraps(Experiment.extract_features)
    def extract_features(self, *args, **kwargs):
        self._proc_scaff(Experiment.extract_features, *args, **kwargs)

    @functools.wraps(Experiment.classify_behaviours)
    def classify_behaviours(self, *args, **kwargs):
        # TODO: handle reading the model file whilst in multiprocessing.
        # Current fix is single processing.
        nprocs = self.nprocs
        self.nprocs = 1
        self._proc_scaff(Experiment.classify_behaviours, *args, **kwargs)
        self.nprocs = nprocs

    @functools.wraps(Experiment.export_behaviours)
    def export_behaviours(self, *args, **kwargs):
        self._proc_scaff(Experiment.export_behaviours, *args, **kwargs)

    @functools.wraps(Experiment.analyse)
    def analyse(self, *args, **kwargs):
        self._proc_scaff(Experiment.analyse, *args, **kwargs)

    @functools.wraps(Experiment.analyse_behaviours)
    def analyse_behaviours(self, *args, **kwargs):
        self._proc_scaff(Experiment.analyse_behaviours, *args, **kwargs)

    @functools.wraps(Experiment.combine_analysis)
    def combine_analysis(self, *args, **kwargs):
        self._proc_scaff(Experiment.combine_analysis, *args, **kwargs)

    @functools.wraps(Experiment.evaluate_vid)
    def evaluate_vid(self, *args, **kwargs):
        # TODO: handle reading the model file whilst in multiprocessing.
        # Current fix is single processing.
        nprocs = self.nprocs
        self.nprocs = 1
        self._proc_scaff(Experiment.evaluate_vid, *args, **kwargs)
        self.nprocs = nprocs

    @functools.wraps(Experiment.export2csv)
    def export2csv(self, *args, **kwargs):
        self._proc_scaff(Experiment.export2csv, *args, **kwargs)

    #####################################################################
    #                CONFIGS DIAGONOSTICS METHODS
    #####################################################################

    def collate_auto_configs(self):
        # Saving the auto fields of the configs of all experiments in the diagnostics folder
        self._proc_scaff(Experiment.collate_auto_configs)
        auto_configs_df = DiagnosticsDf.read(
            os.path.join(self.root_dir, DIAGNOSTICS_DIR, f"{Experiment.collate_auto_configs.__name__}.csv")
        )
        # Making and saving histogram plots of the numerical auto fields
        # NOTE: NOT including string frequencies, only numerical
        auto_configs_df = auto_configs_df.loc[:, auto_configs_df.apply(pd.api.types.is_numeric_dtype)]
        g = sns.FacetGrid(data=auto_configs_df.fillna(-1).melt(), col="variable", sharex=False, col_wrap=4)
        g.map(sns.histplot, "value", bins=10)
        g.set_titles("{col_name}")
        g.savefig(os.path.join(self.root_dir, DIAGNOSTICS_DIR, "collate_auto_configs.png"))
        g.figure.clf()

    #####################################################################
    #            COMBINING ANALYSIS DATA ACROSS EXPS METHODS
    #####################################################################

    def collate_analysis(self) -> None:
        """
        Combines an analysis of all the experiments together to generate combined files for:
        - Each binned data. The index is (bin) and columns are (expName, indiv, measure).
        - The summary data. The index is (expName, indiv, measure) and columns are
        (statistics -e.g., mean).
        """
        # TODO: fix up
        self._analyse_collate_binned()
        self._analyse_collate_summary()

    def _analyse_collate_binned(self) -> None:
        """
        Combines an analysis of all the experiments together to generate combined h5 files for:
        - Each binned data. The index is (bin) and columns are (expName, indiv, measure).
        """
        # Initialising the process and logging description
        description = "Combining binned analysis"
        self.logger.info("%s...", description)
        # AGGREGATING BINNED DATA
        # NOTE: need a more robust way of getting the list of bin sizes
        proj_analyse_dir = os.path.join(self.root_dir, ANALYSIS_DIR)
        configs = ExperimentConfigs.read_json(self.get_experiments()[0].get_fp(Folders.CONFIGS.value))
        bin_sizes_sec = configs.get_ref(configs.user.analyse.bins_sec)
        bin_sizes_sec = np.append(bin_sizes_sec, "custom")
        # Searching through all the analysis subdir
        for analyse_subdir in os.listdir(proj_analyse_dir):
            for bin_i in bin_sizes_sec:
                df_ls = []
                names_ls = []
                for exp in self.get_experiments():
                    in_fp = os.path.join(
                        proj_analyse_dir, analyse_subdir, f"binned_{bin_i}", f"{exp.name}.{AnalyseBinnedDf.IO}"
                    )
                    if os.path.isfile(in_fp):
                        df_ls.append(AnalyseBinnedDf.read(in_fp))
                        names_ls.append(exp.name)
                # Concatenating total_df with df across columns, with experiment name to column MultiIndex
                if len(df_ls) > 0:
                    df = pd.concat(df_ls, keys=names_ls, names=["experiment"], axis=1)
                    out_fp = os.path.join(
                        proj_analyse_dir,
                        analyse_subdir,
                        f"__ALL_binned_{bin_i}.{AnalyseBinnedDf.IO}",
                    )
                    AnalyseBinnedDf.write(df, out_fp)

    def _analyse_collate_summary(self) -> None:
        """
        Combines an analysis of all the experiments together to generate combined h5 files for:
        - The summary data. The index is (expName, indiv, measure) and columns are
        (statistics -e.g., mean).
        """
        # Initialising the process and logging description
        description = "Combining summary analysis"
        self.logger.info("%s...", description)
        # AGGREGATING SUMMARY DATA
        proj_analyse_dir = os.path.join(self.root_dir, ANALYSIS_DIR)
        # Searching through all the analysis subdir
        for analyse_subdir in os.listdir(proj_analyse_dir):
            df_ls = []
            names_ls = []
            for exp in self.get_experiments():
                in_fp = os.path.join(proj_analyse_dir, analyse_subdir, "summary", f"{exp.name}.{AnalyseSummaryDf.IO}")
                if os.path.isfile(in_fp):
                    # Reading exp summary df
                    df_ls.append(AnalyseSummaryDf.read(in_fp))
                    names_ls.append(exp.name)
            out_fp = os.path.join(proj_analyse_dir, analyse_subdir, f"__ALL_summary.{AnalyseSummaryDf.IO}")
            # Concatenating total_df with df across columns, with experiment name to column MultiIndex
            if len(df_ls) > 0:
                total_df = pd.concat(df_ls, keys=names_ls, names=["experiment"], axis=0)
                AnalyseSummaryDf.write(total_df, out_fp)
