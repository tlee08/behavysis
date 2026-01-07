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

from behavysis.constants import (
    ANALYSIS_DIR,
    DIAGNOSTICS_DIR,
    Folders,
)
from behavysis.df_classes.analysis_agg_df import AnalysisBinnedDf, AnalysisSummaryDf
from behavysis.df_classes.analysis_collated_df import AnalysisBinnedCollatedDf, AnalysisSummaryCollatedDf
from behavysis.df_classes.diagnostics_df import DiagnosticsDf
from behavysis.pipeline.experiment import Experiment
from behavysis.processes.run_dlc import RunDLC
from behavysis.models.experiment_configs import (
    ExperimentConfigs,
)
from behavysis.utils.dask_utils import cluster_process
from behavysis.utils.io_utils import get_name
from behavysis.utils.logging_utils import init_logger_file
from behavysis.utils.multiproc_utils import get_gpu_ids


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

    logger = init_logger_file()

    root_dir: str
    _experiments: dict[str, Experiment]
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
        self._experiments = {}
        self.nprocs = 4

    #####################################################################
    # GETTER METHODS
    #####################################################################

    @property
    def experiments(self) -> list[Experiment]:
        """
        Gets the ordered list of Experiment instances in the Project.

        Returns
        -------
        list[Experiment]
            The list of all Experiment instances stored in the Project instance.
        """
        return [self._experiments[i] for i in natsorted(self._experiments)]

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
        if name in self._experiments:
            return self._experiments[name]
        raise ValueError(f'Experiment with the name "{name}" does not exist in the project.')

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
        with cluster_process(LocalCluster(n_workers=self.nprocs, threads_per_worker=1)):
            # Preparing all experiments for execution
            f_d_ls = [dask.delayed(method)(exp, *args, **kwargs) for exp in self.experiments]  # type: ignore
            # Executing in parallel
            dd_ls = list(dask.compute(*f_d_ls))  # type: ignore
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
        return [method(exp, *args, **kwargs) for exp in self.experiments]

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
        self.logger.info(f"Running {method.__name__} for all experiments.")
        # Running
        dd_ls = scaffold_func(method, *args, **kwargs)
        if len(dd_ls) > 0:
            # Processing all experiments
            df = DiagnosticsDf.init_from_dd_ls(dd_ls)
            # Updating the diagnostics file at each step
            DiagnosticsDf.write(df, os.path.join(self.root_dir, DIAGNOSTICS_DIR, f"{method.__name__}.csv"))
            # Finishing
            self.logger.info(f"Finished running {method.__name__} for all experiments")

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
        if name not in self._experiments:
            self._experiments[name] = Experiment(name, self.root_dir)
            return True
        return False

    def import_experiments(self) -> None:
        """
        Add all experiments in the project folder to the experiments dict.
        The key of each experiment in the .experiments dict is "name".
        Refer to Project.addExperiment() for details about how each experiment is added.
        """
        self.logger.info(f"Searching project folder: {self.root_dir}")
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
            for fp_name in natsorted(os.listdir(folder)):
                if re.search(r"^\.", fp_name):  # do not add hidden files
                    continue
                name = get_name(fp_name)
                try:
                    self.import_experiment(name)
                    dd_dict[f.value].append(name)
                except ValueError as e:  # do not add invalid files
                    self.logger.info(f"failed: {f.value}    --    {fp_name}: {e}")
        # Logging outcome of imported and failed experiments
        exp_ls_msg = "".join([f"\n    - {exp.name}" for exp in self.experiments])
        self.logger.info(f"Experiments imported:{exp_ls_msg}")
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

    def update_configs(self, default_configs_fp: str, overwrite: str) -> None:
        self._proc_scaff(Experiment.update_configs, default_configs_fp, overwrite)

    def format_vid(self, overwrite: bool) -> None:
        self._proc_scaff(Experiment.format_vid, overwrite)

    def get_vid_metadata(self) -> None:
        self._proc_scaff(Experiment.get_vid_metadata)

    def run_dlc(self, gputouse: int | None = None, overwrite: bool = False) -> None:
        """
        Batch processing corresponding to
        [behavysis.pipeline.experiment.Experiment.run_dlc][]

        Uses a multiprocessing pool to run DLC on each batch of experiments with each GPU
        natively as batch in the same spawned subprocess (a DLC subprocess is spawned).
        This is a slight tweak from the regular method of running
        each experiment separately with multiprocessing.
        """
        # TODO: implement diagnostics
        # TODO: implement error handling
        # TODO: implement handling if NO GPU (i.e. nprocs == 0)
        # If gputouse is not specified, using all GPUs
        gputouse_ls = get_gpu_ids() if gputouse is None else [gputouse]
        nprocs = len(gputouse_ls)
        # Getting the experiments to run DLC on
        exp_ls = self.experiments
        # If overwrite is False, filtering for only experiments that need processing
        if not overwrite:
            exp_ls = [exp for exp in exp_ls if not os.path.isfile(exp.get_fp(Folders.KEYPOINTS.value))]
        # Running DLC on each batch of experiments with each GPU (given allocated GPU ID)
        exp_batches_ls = np.array_split(np.array(exp_ls), nprocs)
        # Starting a dask cluster
        with cluster_process(LocalCluster(n_workers=nprocs, threads_per_worker=1)):
            # Preparing all experiments for execution
            f_d_ls = [
                dask.delayed(RunDLC.ma_dlc_run_batch)(  # type: ignore
                    vid_fp_ls=[exp.get_fp(Folders.FORMATTED_VID.value) for exp in exp_batch],
                    keypoints_dir=os.path.join(self.root_dir, Folders.KEYPOINTS.value),
                    configs_dir=os.path.join(self.root_dir, Folders.CONFIGS.value),
                    gputouse=gputouse,
                    overwrite=overwrite,
                )
                for gputouse, exp_batch in zip(gputouse_ls, exp_batches_ls)
            ]
            # Executing in parallel
            list(dask.compute(*f_d_ls))  # type: ignore

    def calculate_parameters(self, funcs: tuple[Callable, ...]) -> None:
        self._proc_scaff(Experiment.calculate_parameters, funcs)

    def collate_auto_configs(self) -> None:
        # Saving the auto fields of the configs of all experiments in the diagnostics folder
        self._proc_scaff(Experiment.collate_auto_configs)
        f_name = Experiment.collate_auto_configs.__name__
        auto_configs_df = DiagnosticsDf.read(os.path.join(self.root_dir, DIAGNOSTICS_DIR, f"{f_name}.csv"))
        # Making and saving histogram plots of the numerical auto fields
        # NOTE: NOT including string frequencies, only numerical
        auto_configs_df = auto_configs_df.loc[:, auto_configs_df.apply(pd.api.types.is_numeric_dtype)]
        g = sns.FacetGrid(
            data=auto_configs_df.fillna(-1).melt(var_name="measure", value_name="value"),
            col="measure",
            sharex=False,
            col_wrap=4,
        )
        g.map(sns.histplot, "value", bins=10)
        g.set_titles("{col_name}")
        g.savefig(os.path.join(self.root_dir, DIAGNOSTICS_DIR, "collate_auto_configs.png"))
        g.figure.clf()

    def preprocess(self, funcs: tuple[Callable, ...], overwrite: bool) -> None:
        self._proc_scaff(Experiment.preprocess, funcs, overwrite)

    def extract_features(self, overwrite: bool) -> None:
        self._proc_scaff(Experiment.extract_features, overwrite)

    def classify_behavs(self, overwrite: bool) -> None:
        # TODO: IO error with multiprocessing. Using single processing for now.
        nprocs = self.nprocs
        self.nprocs = 1
        self._proc_scaff(Experiment.classify_behavs, overwrite)
        self.nprocs = nprocs

    def export_behavs(self, overwrite: bool) -> None:
        self._proc_scaff(Experiment.export_behavs, overwrite)

    def analyse(self, funcs: tuple[Callable, ...]) -> None:
        self._proc_scaff(Experiment.analyse, funcs)

    def analyse_behavs(self) -> None:
        self._proc_scaff(Experiment.analyse_behavs)

    def combine_analysis(self) -> None:
        self._proc_scaff(Experiment.combine_analysis)

    def evaluate_vid(self, overwrite: bool) -> None:
        # TODO: IO error with multiprocessing. Using single processing for now.
        # nprocs = self.nprocs
        # self.nprocs = 1
        self._proc_scaff(Experiment.evaluate_vid, overwrite)
        # self.nprocs = nprocs

    @functools.wraps(Experiment.export2csv)
    def export2csv(self, src_dir: str, dst_dir: str, overwrite: bool) -> None:
        self._proc_scaff(Experiment.export2csv, src_dir, dst_dir, overwrite)

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
        configs = ExperimentConfigs.read_json(self.experiments[0].get_fp(Folders.CONFIGS.value))
        bin_sizes_sec = configs.get_ref(configs.user.analyse.bins_sec)
        bin_sizes_sec = np.append(bin_sizes_sec, "custom")
        # Checking that the analysis directory exists
        if not os.path.isdir(proj_analyse_dir):
            return
        # Searching through all the analysis subdir
        for analyse_subdir in os.listdir(proj_analyse_dir):
            for bin_i in bin_sizes_sec:
                df_ls = []
                names_ls = []
                for exp in self.experiments:
                    in_fp = os.path.join(
                        proj_analyse_dir, analyse_subdir, f"binned_{bin_i}", f"{exp.name}.{AnalysisBinnedDf.IO}"
                    )
                    if os.path.isfile(in_fp):
                        df_ls.append(AnalysisBinnedDf.read(in_fp))
                        names_ls.append(exp.name)
                # Concatenating total_df with df across columns, with experiment name to column MultiIndex
                if len(df_ls) > 0:
                    df = pd.concat(df_ls, keys=names_ls, names=["experiment"], axis=1)
                    df = df.fillna(0)
                    AnalysisBinnedCollatedDf.write(
                        df,
                        os.path.join(
                            proj_analyse_dir, analyse_subdir, f"__ALL_binned_{bin_i}.{AnalysisBinnedCollatedDf.IO}"
                        ),
                    )
                    AnalysisBinnedCollatedDf.write_csv(
                        df, os.path.join(proj_analyse_dir, analyse_subdir, f"__ALL_binned_{bin_i}.csv")
                    )

    def _analyse_collate_summary(self) -> None:
        """
        Combines an analysis of all the experiments together to generate combined h5 files for:
        - The summary data. The index is (expName, indiv, measure) and columns are
        (statistics -e.g., mean).
        
        Dataframe structure:
        - Rows: single index with experiment
        - Columns MultiIndex with each different measure and summary statistic
        """
        # Initialising the process and logging description
        description = "Combining summary analysis"
        self.logger.info("%s...", description)
        # AGGREGATING SUMMARY DATA
        proj_analyse_dir = os.path.join(self.root_dir, ANALYSIS_DIR)
        # Checking that the analysis directory exists
        if not os.path.isdir(proj_analyse_dir):
            return
        # Searching through all the analysis subdir
        for analyse_subdir in os.listdir(proj_analyse_dir):
            df_ls = []
            names_ls = []
            for exp in self.experiments:
                in_fp = os.path.join(proj_analyse_dir, analyse_subdir, "summary", f"{exp.name}.{AnalysisSummaryDf.IO}")
                if os.path.isfile(in_fp):
                    # Reading exp summary df
                    df_i = AnalysisSummaryDf.read(in_fp)
                    # Converting to (ideally) a pd.Series with MultiIndex columns
                    df_i = df_i.stack()
                    # Appending for concatenation
                    df_ls.append(df_i)
                    names_ls.append(exp.name)
            # Concatenating total_df with df across columns, with experiment name to column MultiIndex
            if len(df_ls) > 0:
                df = pd.concat(df_ls, keys=names_ls, names=["experiment"], axis=0)
                df = df.fillna(0)
                AnalysisSummaryCollatedDf.write(
                    df, os.path.join(proj_analyse_dir, analyse_subdir, f"__ALL_summary.{AnalysisSummaryCollatedDf.IO}")
                )
                AnalysisSummaryCollatedDf.write_csv(
                    df, os.path.join(proj_analyse_dir, analyse_subdir, "__ALL_summary.csv")
                )
