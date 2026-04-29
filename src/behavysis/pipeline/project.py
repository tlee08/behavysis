"""Project class for batch processing multiple experiments."""

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dask
import numpy as np
import pandas as pd
import seaborn as sns
from dask.distributed import LocalCluster
from natsort import natsorted

from behavysis.constants import (
    ANALYSIS_DIR,
    CACHE_DIR,
    DIAGNOSTICS_DIR,
    Folders,
)
from behavysis.df_classes.analysis_agg_df import AnalysisBinnedDf, AnalysisSummaryDf
from behavysis.df_classes.analysis_collated_df import (
    AnalysisBinnedCollatedDf,
    AnalysisSummaryCollatedDf,
)
from behavysis.df_classes.diagnostics_df import DiagnosticsDf
from behavysis.models.experiment_configs import ExperimentConfigs
from behavysis.pipeline.experiment import Experiment
from behavysis.processes.run_dlc import RunDLC
from behavysis.utils.dask_utils import cluster_process
from behavysis.utils.io_utils import get_name
from behavysis.utils.logging_utils import setup_logging
from behavysis.utils.multiproc_utils import get_gpu_ids

logger = logging.getLogger(__name__)


class Project:
    """A project is used to process and analyse many experiments at the same time."""

    root_dir: Path
    _experiments: dict[str, Experiment]
    nprocs: int

    def __init__(self, root_dir: str | Path) -> None:
        root_dir = Path(root_dir)
        if not root_dir.is_dir():
            msg = f'The folder "{root_dir}" does not exist.'
            raise ValueError(msg)
        self.root_dir = root_dir.resolve()
        self._experiments = {}
        self.nprocs = 4
        project_name = self.root_dir.name
        setup_logging(log_file=CACHE_DIR / project_name)

    @property
    def experiments(self) -> list[Experiment]:
        """Gets the ordered list of Experiment instances."""
        return [self._experiments[i] for i in natsorted(self._experiments)]

    def get_experiment(self, name: str) -> Experiment:
        if name in self._experiments:
            return self._experiments[name]
        msg = f'Experiment "{name}" does not exist in the project.'
        raise ValueError(msg)

    def _run_parallel(self, method: Callable, *args: Any, **kwargs: Any) -> list[Any]:
        """Run a method on all experiments in parallel."""
        with cluster_process(LocalCluster(n_workers=self.nprocs, threads_per_worker=1)):
            delayed_tasks = [
                dask.delayed(method)(exp, *args, **kwargs) for exp in self.experiments
            ]
            return list(dask.compute(*delayed_tasks))

    def _run_sequential(self, method: Callable, *args: Any, **kwargs: Any) -> list[Any]:
        """Run a method on all experiments sequentially."""
        return [method(exp, *args, **kwargs) for exp in self.experiments]

    def _run_and_save_diagnostics(
        self, method: Callable, *args: Any, **kwargs: Any
    ) -> None:
        """Run a method on all experiments and save diagnostics."""
        logger.info(f"Running {method.__name__} for all experiments.")
        runner = self._run_sequential if self.nprocs == 1 else self._run_parallel
        results = runner(method, *args, **kwargs)
        if not results:
            return

        diagnostics_dir = self.root_dir / DIAGNOSTICS_DIR
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        json_path = diagnostics_dir / f"{method.__name__}.json"
        with open(json_path, "w") as f:
            json.dump([r.model_dump() for r in results], f, indent=2, default=str)

        logger.info(f"Finished running {method.__name__} for all experiments")

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
                name = get_name(fp_name.name)
                try:
                    self.import_experiment(name)
                    dd_dict[f.value].append(name)
                except ValueError as e:
                    logger.info(f"Failed: {f.value} - {fp_name.name}: {e}")
        exp_ls_msg = "".join([f"\n    - {exp.name}" for exp in self.experiments])
        logger.info(f"Experiments imported:{exp_ls_msg}")
        all_names = np.unique(np.concatenate(list(dd_dict.values())))
        df = pd.DataFrame(index=all_names)
        for folder, names in dd_dict.items():
            df[folder] = df.index.isin(names)
        DiagnosticsDf.write(
            df,
            self.root_dir / DIAGNOSTICS_DIR / "import_experiments.csv",
        )

    def update_configs(self, default_configs_fp: Path, overwrite: str) -> None:
        self._run_and_save_diagnostics(
            Experiment.update_configs, default_configs_fp, overwrite
        )

    def format_vid(self, *, overwrite: bool) -> None:
        self._run_and_save_diagnostics(Experiment.format_vid, overwrite=overwrite)

    def get_vid_metadata(self) -> None:
        self._run_and_save_diagnostics(Experiment.get_vid_metadata)

    def run_dlc(self, gputouse: int | None = None, overwrite: bool = False) -> None:
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
                dask.delayed(RunDLC.ma_dlc_run_batch)(
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
        self._run_and_save_diagnostics(Experiment.calculate_parameters, funcs)

    def collate_auto_configs(self) -> None:
        self._run_and_save_diagnostics(Experiment.collate_auto_configs)
        json_path = self.root_dir / DIAGNOSTICS_DIR / "collate_auto_configs.json"
        with open(json_path) as f:
            results = json.load(f)
        records = [
            r["results"].get("data", {}) for r in results if r["results"].get("data")
        ]
        if not records:
            return
        df = pd.DataFrame(records)
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            return
        df = df[numeric_cols].fillna(-1).melt(var_name="measure", value_name="value")
        g = sns.FacetGrid(data=df, col="measure", sharex=False, col_wrap=4)
        g.map(sns.histplot, "value", bins=10)
        g.set_titles("{col_name}")
        g.savefig(self.root_dir / DIAGNOSTICS_DIR / "collate_auto_configs.png")
        g.figure.clf()

    def preprocess(self, funcs: tuple[Callable, ...], *, overwrite: bool) -> None:
        self._run_and_save_diagnostics(
            Experiment.preprocess, funcs, overwrite=overwrite
        )

    def extract_features(self, *, overwrite: bool) -> None:
        self._run_and_save_diagnostics(Experiment.extract_features, overwrite=overwrite)

    def classify_behavs(self, *, overwrite: bool) -> None:
        # Temporarily use single processing due to IO issues
        nprocs = self.nprocs
        self.nprocs = 1
        self._run_and_save_diagnostics(Experiment.classify_behavs, overwrite=overwrite)
        self.nprocs = nprocs

    def export_behavs(self, *, overwrite: bool) -> None:
        self._run_and_save_diagnostics(Experiment.export_behavs, overwrite=overwrite)

    def analyse(self, funcs: tuple[Callable, ...]) -> None:
        self._run_and_save_diagnostics(Experiment.analyse, funcs)

    def analyse_behavs(self) -> None:
        self._run_and_save_diagnostics(Experiment.analyse_behavs)

    def combine_analysis(self) -> None:
        self._run_and_save_diagnostics(Experiment.combine_analysis)

    def evaluate_vid(self, *, overwrite: bool) -> None:
        self._run_and_save_diagnostics(Experiment.evaluate_vid, overwrite=overwrite)

    def export2csv(self, src_dir: str, dst_dir: str | Path, *, overwrite: bool) -> None:
        self._run_and_save_diagnostics(
            Experiment.export2csv, src_dir, dst_dir, overwrite=overwrite
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
                AnalysisBinnedCollatedDf.write_csv(
                    df, subdir / f"__ALL_binned_{bin_size}.csv"
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
            AnalysisSummaryCollatedDf.write_csv(df, subdir / "__ALL_summary.csv")
