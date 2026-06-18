"""Project class for batch processing multiple experiments."""

import re
from collections.abc import Callable
from pathlib import Path

import dask
import numpy as np
import pandas as pd
from dask.distributed import LocalCluster
from loguru import logger
from natsort import natsorted

from behavysis.constants import (
    ANALYSIS_DIR,
    Folders,
)
from behavysis.df_classes.analysis_agg_df import AnalysisBinnedDf, AnalysisSummaryDf
from behavysis.df_classes.analysis_collated_df import (
    AnalysisBinnedCollatedDf,
    AnalysisSummaryCollatedDf,
)
from behavysis.models.experiment_configs import ExperimentConfigs
from behavysis.pipeline.experiment import Experiment
from behavysis.processes.run_dlc import ma_dlc_run_batch
from behavysis.utils.dask_utils import cluster_process
from behavysis.utils.multiproc_utils import get_gpu_ids


class Project:
    """A project is used to process and analyse many experiments at the same time."""

    root_dir: Path
    _experiments: dict[str, Experiment]
    nprocs: int

    def __init__(self, root_dir: str | Path) -> None:
        """Initialize a Project instance."""
        root_dir = Path(root_dir)
        if not root_dir.is_dir():
            msg = (
                f'Project folder not found: "{root_dir}"\n'
                f"  Create a new project with: behavysis-make-project"
            )
            raise ValueError(msg)
        self.root_dir = root_dir.resolve()
        self._experiments = {}
        self.nprocs = 4

    @property
    def experiments(self) -> list[Experiment]:
        """Gets the ordered list of Experiment instances."""
        return [self._experiments[i] for i in natsorted(self._experiments)]

    def get_experiment(self, name: str) -> Experiment:
        """Get an experiment by name."""
        if name in self._experiments:
            return self._experiments[name]
        msg = (
            f'Experiment "{name}" not found in project.\n'
            f"  Did you forget to call proj.import_experiments()?"
        )
        raise ValueError(msg)

    def _run_parallel(
        self,
        method: Callable[..., None],
        **kwargs,
    ) -> None:
        """Run a method on all experiments in parallel."""
        with cluster_process(LocalCluster(n_workers=self.nprocs, threads_per_worker=1)):
            delayed_tasks = [
                dask.delayed(method)(exp, **kwargs) for exp in self.experiments
            ]
            dask.compute(*delayed_tasks)

    def _run_sequential(
        self,
        method: Callable[..., None],
        **kwargs,
    ) -> None:
        """Run a method on all experiments sequentially."""
        for exp in self.experiments:
            method(exp, **kwargs)

    def _run(
        self,
        _func: Callable[..., None],
        **kwargs,
    ) -> None:
        """Run a method on all experiments."""
        runner = self._run_sequential if self.nprocs == 1 else self._run_parallel
        runner(_func, **kwargs)

    def import_experiment(self, name: str) -> bool:
        """Add an experiment to the project. Returns True if newly added."""
        if name not in self._experiments:
            self._experiments[name] = Experiment(name, self.root_dir)
            return True
        return False

    def import_experiments(self) -> None:
        """Import all experiments from the project folder."""
        logger.info(f"Searching project folder: {self.root_dir}")
        dd_dict: dict[str, list[str]] = {}
        for f in Folders:
            folder = self.root_dir / f.value
            dd_dict[f.value] = []
            if not folder.is_dir():
                continue
            for fp_name in natsorted(folder.iterdir()):
                if re.search(r"^\.", fp_name.name):
                    continue
                name = fp_name.stem
                try:
                    self.import_experiment(name)
                    dd_dict[f.value].append(name)
                except ValueError as e:
                    logger.info(f"Failed: {f.value} - {fp_name.name}: {e}")
        exp_ls_msg = "".join([f"\n    - {exp.name}" for exp in self.experiments])
        logger.info(f"Experiments imported:{exp_ls_msg}")

    def update_configs(self, default_configs_fp: Path, *, overwrite: str) -> None:
        """Update the configs for all experiments."""
        self._run(
            Experiment.update_configs,
            default_configs_fp=default_configs_fp,
            overwrite=overwrite,
        )

    def format_vid(self, *, overwrite: bool) -> None:
        """Format videos for all experiments."""
        self._run(
            Experiment.format_vid,
            overwrite=overwrite,
        )

    def run_dlc(self, gputouse: int | None = None, *, overwrite: bool = False) -> None:
        """Run DLC on all experiments with GPU batching."""
        gputouse_ls = get_gpu_ids() if gputouse is None else [gputouse]
        nprocs = len(gputouse_ls)
        exp_ls = self.experiments
        if not overwrite:
            exp_ls = [
                exp for exp in exp_ls if not exp.get_fp(Folders.KEYPOINTS).is_file()
            ]
        if not exp_ls:
            return
        exp_batches = np.array_split(np.array(exp_ls), nprocs)
        with cluster_process(LocalCluster(n_workers=nprocs, threads_per_worker=1)):
            delayed_tasks = [
                dask.delayed(ma_dlc_run_batch)(
                    vid_fp_ls=[exp.get_fp(Folders.FORMATTED_VID) for exp in batch],
                    keypoints_dir=self.root_dir / Folders.KEYPOINTS.value,
                    configs_dir=self.root_dir / Folders.CONFIGS.value,
                    gputouse=gpu,
                    overwrite=overwrite,
                )
                for gpu, batch in zip(gputouse_ls, exp_batches, strict=False)
            ]
            list(dask.compute(*delayed_tasks))

    def calculate_parameters(self, funcs: tuple[Callable, ...]) -> None:
        """Calculate parameters for all experiments."""
        self._run(
            Experiment.calculate_parameters,
            funcs=funcs,
        )

    def preprocess(self, funcs: tuple[Callable, ...], *, overwrite: bool) -> None:
        """Preprocess all experiments."""
        self._run(
            Experiment.preprocess,
            funcs=funcs,
            overwrite=overwrite,
        )

    def extract_features(self, *, overwrite: bool) -> None:
        """Extract features for all experiments."""
        self._run(
            Experiment.extract_features,
            overwrite=overwrite,
        )

    def classify_behavs(self, *, overwrite: bool) -> None:
        """Classify behaviours for all experiments."""
        # Temporarily use single processing due to IO issues
        nprocs = self.nprocs
        self.nprocs = 1
        self._run(
            Experiment.classify_behavs,
            overwrite=overwrite,
        )
        self.nprocs = nprocs

    def export_behavs(self, *, overwrite: bool) -> None:
        """Export predicted behaviours for all experiments."""
        self._run(
            Experiment.export_behavs,
            overwrite=overwrite,
        )

    def analyse(self, funcs: tuple[Callable, ...]) -> None:
        """Analyse all experiments."""
        self._run(
            Experiment.analyse,
            funcs=funcs,
        )

    def analyse_behavs(self) -> None:
        """Analyse behaviours for all experiments."""
        self._run(
            Experiment.analyse_behavs,
        )

    def combine_analysis(self) -> None:
        """Combine analysis for all experiments."""
        self._run(
            Experiment.combine_analysis,
        )

    def evaluate_vid(self, *, overwrite: bool) -> None:
        """Evaluate videos for all experiments."""
        self._run(
            Experiment.evaluate_vid,
            overwrite=overwrite,
        )

    def export2csv(self, src_dir: str, dst_dir: str | Path, *, overwrite: bool) -> None:
        """Export dataframe to CSV for all experiments."""
        self._run(
            Experiment.export2csv,
            src_dir=src_dir,
            dst_dir=dst_dir,
            overwrite=overwrite,
        )

    def collate_analysis(self) -> None:
        """Combine analysis across all experiments."""
        self._collate_binned()
        self._collate_summary()

    def _collate_binned(self) -> None:
        """Combine binned analysis data across experiments."""
        logger.info("Collating binned analysis...")
        proj_analyse_dir = self.root_dir / ANALYSIS_DIR
        if not proj_analyse_dir.is_dir():
            return

        configs = ExperimentConfigs.model_validate_json(
            self.experiments[0].get_fp(Folders.CONFIGS).read_text()
        )
        bin_sizes = [*list(configs.get_ref(configs.user.analyse.bins_sec)), "custom"]

        for subdir in proj_analyse_dir.iterdir():
            if not subdir.is_dir():
                continue
            for bin_size in bin_sizes:
                df_ls, names_ls = [], []
                for exp in self.experiments:
                    in_fp = (
                        subdir
                        / f"binned_{bin_size}"
                        / f"{exp.name}.{AnalysisBinnedDf.IO}"
                    )
                    if in_fp.is_file():
                        df_ls.append(AnalysisBinnedDf.read(in_fp))
                        names_ls.append(exp.name)
                if not df_ls:
                    continue
                df = pd.concat(
                    df_ls, keys=names_ls, names=["experiment"], axis=1
                ).fillna(0)
                AnalysisBinnedCollatedDf.write(
                    df,
                    subdir / f"__ALL_binned_{bin_size}.{AnalysisBinnedCollatedDf.IO}",
                )
                AnalysisBinnedCollatedDf.write(
                    df, subdir / f"__ALL_binned_{bin_size}.csv", fmt="csv"
                )

    def _collate_summary(self) -> None:
        """Combine summary analysis data across experiments."""
        logger.info("Collating summary analysis...")
        proj_analyse_dir = self.root_dir / ANALYSIS_DIR
        if not proj_analyse_dir.is_dir():
            return

        for subdir in proj_analyse_dir.iterdir():
            if not subdir.is_dir():
                continue
            df_ls, names_ls = [], []
            for exp in self.experiments:
                in_fp = subdir / "summary" / f"{exp.name}.{AnalysisSummaryDf.IO}"
                if in_fp.is_file():
                    df_ls.append(AnalysisSummaryDf.read(in_fp))
                    names_ls.append(exp.name)
            if not df_ls:
                continue
            df = pd.concat(df_ls, keys=names_ls, names=["experiment"], axis=0).fillna(0)
            AnalysisSummaryCollatedDf.write(
                df, subdir / f"__ALL_summary.{AnalysisSummaryCollatedDf.IO}"
            )
            AnalysisSummaryCollatedDf.write(df, subdir / "__ALL_summary.csv", fmt="csv")
