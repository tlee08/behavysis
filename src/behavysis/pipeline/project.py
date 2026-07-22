"""Project class for batch processing multiple experiments."""

from collections.abc import Callable
from pathlib import Path

import dask
import numpy as np
import polars as pl
from dask.distributed import LocalCluster
from loguru import logger
from natsort import natsorted

from behavysis.constants import (
    AGG,
    ANALYSIS_DIR,
    BIN_SEC,
    BINNED,
    DF_IO_FORMAT,
    EXPERIMENT,
    FORMATTED_VIDEO_DIR,
    INDIVIDUAL,
    KEYPOINTS_DIR,
    MEASURE,
    SUMMARY,
    VALUE,
)
from behavysis.funcs import (
    AnalyseFunc,
    CalculateParametersFunc,
    ExtractFeaturesFunc,
    PreprocessFunc,
)
from behavysis.funcs.run_dlc import ma_dlc_run_batch
from behavysis.pipeline import Experiment
from behavysis.schemas import read_df, write_df
from behavysis.utils import cluster_process, get_gpu_device_ids, pass_exception


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
        **kwargs: object,
    ) -> None:
        """Run a method on all experiments in parallel."""
        with cluster_process(LocalCluster(n_workers=self.nprocs, threads_per_worker=1)):
            delayed_tasks = [
                dask.delayed(pass_exception(method))(exp, **kwargs)
                for exp in self.experiments
            ]
            dask.compute(*delayed_tasks)

    def _run_sequential(
        self,
        method: Callable[..., None],
        **kwargs: object,
    ) -> None:
        """Run a method on all experiments sequentially."""
        for exp in self.experiments:
            pass_exception(method)(exp, **kwargs)

    def _run(
        self,
        _func: Callable[..., None],
        **kwargs: object,
    ) -> None:
        """Run a method on all experiments."""
        runner = self._run_sequential if self.nprocs == 1 else self._run_parallel
        runner(_func, **kwargs)

    def import_experiments(self, name_ls: list[str]) -> None:
        """Import all experiments in a given list.

        Expects list to be names without suffix. E.g:
        ["exp1", "exp2", ...]
        """
        logger.info(f"Searching project folder: {self.root_dir}")
        for name in natsorted(name_ls):
            try:
                self._experiments[name] = Experiment(name, self.root_dir)
            except ValueError as e:
                logger.info(f"Failed: {name}: {e}")
        exp_ls_msg = "".join([f"\n    - {exp.name}" for exp in self.experiments])
        logger.info(f"Experiments imported:{exp_ls_msg}")

    def update_config(self, default_config_fp: Path) -> None:
        """Update the config for all experiments."""
        self._run(
            Experiment.update_config,
            default_config_fp=default_config_fp,
        )

    def format_video(self, *, overwrite: bool) -> None:
        """Format videos for all experiments."""
        self._run(
            Experiment.format_video,
            overwrite=overwrite,
        )

    def get_video_metadata(self) -> None:
        """Get video metadata and save."""
        self._run(
            Experiment.get_video_metadata,
        )

    def run_dlc(self, gputouse: int | None = None, *, overwrite: bool) -> None:
        """Run DLC on all experiments with GPU batching."""
        gputouse_ls = get_gpu_device_ids() if gputouse is None else [gputouse]
        nprocs = len(gputouse_ls)
        # Get list of experiment to run
        # Consider overwrite flag and if keypoints dfs exist
        exp_ls = self.experiments
        if not overwrite:
            exp_ls = [exp for exp in exp_ls if not exp.get_fp(KEYPOINTS_DIR).is_file()]
        if not exp_ls:
            return
        # Validating that all dlc config_yaml_fp are the same
        dlc_config_fp_set = {
            exp.read_config().require_run_dlc().model_fp for exp in exp_ls
        }
        if len(dlc_config_fp_set) != 1:
            logger.warning("All experiments must have the same DLC config file")
            return
        dlc_config_fp = dlc_config_fp_set.pop()
        # Splitting into nprocs batches
        exp_batches = np.array_split(np.array(exp_ls), nprocs)
        # Running DLC
        with cluster_process(LocalCluster(n_workers=nprocs, threads_per_worker=1)):
            delayed_tasks = [
                dask.delayed(ma_dlc_run_batch)(
                    vid_fp_ls=[exp.get_fp(FORMATTED_VIDEO_DIR) for exp in batch],
                    keypoints_dir=self.root_dir / KEYPOINTS_DIR,
                    dlc_config_fp=dlc_config_fp,
                    gputouse=gpu,
                )
                for gpu, batch in zip(gputouse_ls, exp_batches, strict=False)
            ]
            list(dask.compute(*delayed_tasks))

    def calculate_parameters(self, funcs: tuple[CalculateParametersFunc, ...]) -> None:
        """Calculate parameters for all experiments."""
        self._run(
            Experiment.calculate_parameters,
            funcs=funcs,
        )

    def preprocess(self, funcs: tuple[PreprocessFunc, ...], *, overwrite: bool) -> None:
        """Preprocess all experiments."""
        self._run(
            Experiment.preprocess,
            funcs=funcs,
            overwrite=overwrite,
        )

    def extract_features(
        self, funcs: tuple[ExtractFeaturesFunc, ...], *, overwrite: bool
    ) -> None:
        """Extract features for all experiments."""
        self._run(
            Experiment.extract_features,
            funcs=funcs,
            overwrite=overwrite,
        )

    def classify_behaviour(self, *, overwrite: bool) -> None:
        """Classify behaviours for all experiments."""
        # Temporarily use single processing due to IO issues
        nprocs = self.nprocs
        self.nprocs = 1
        self._run(
            Experiment.classify_behaviour,
            overwrite=overwrite,
        )
        self.nprocs = nprocs

    def export_behaviour(self, *, overwrite: bool) -> None:
        """Export predicted behaviours for all experiments."""
        self._run(
            Experiment.export_behaviour,
            overwrite=overwrite,
        )

    def analyse(self, funcs: tuple[AnalyseFunc, ...]) -> None:
        """Analyse all experiments."""
        self._run(
            Experiment.analyse,
            funcs=funcs,
        )

    def analyse_behaviour(self) -> None:
        """Analyse behaviours for all experiments."""
        self._run(
            Experiment.analyse_behaviour,
        )

    def combine_analysis(self) -> None:
        """Combine analysis for all experiments."""
        self._run(
            Experiment.combine_analysis,
        )

    def collate_analysis(self) -> None:
        """Combine analysis across all experiments."""
        logger.info("Collating binned analysis...")
        proj_analyse_dir = self.root_dir / ANALYSIS_DIR
        if not proj_analyse_dir.is_dir():
            return
        # For each subdir, loop throught dirs in that subdir to
        # combine dataframes
        for subdir1 in proj_analyse_dir.iterdir():
            if not subdir1.is_dir():
                continue
            for subdir2 in subdir1.iterdir():
                df_ls = []
                for exp in self.experiments:
                    # Construct filepath for experiment's analysis
                    in_fp = subdir2 / f"{exp.name}.{DF_IO_FORMAT}"
                    if in_fp.is_file():
                        # Read
                        df = read_df(in_fp)
                        # Add experiment column
                        df = df.with_columns(pl.lit(exp.name).alias(EXPERIMENT)).select(
                            pl.col(EXPERIMENT), pl.all().exclude(EXPERIMENT)
                        )
                        # Append to list
                        df_ls.append(df)
                # Skip if no data
                if not df_ls:
                    continue
                # Concatenate data
                combined_df = pl.concat(df_ls)
                # Write Parquet
                out_fp = subdir1 / f"all_{subdir2.stem}.{DF_IO_FORMAT}"
                write_df(combined_df, out_fp)
                # Also write CSV, which has been formatted to wider
                # (easier to view)
                combined_csv_df = combined_df.to_pandas()
                if BINNED in subdir2.stem:
                    # If binned
                    _cols = [INDIVIDUAL, BIN_SEC, MEASURE, AGG]
                    combined_csv_df = combined_csv_df.set_index([EXPERIMENT, *_cols])[
                        VALUE
                    ].unstack(_cols)
                elif SUMMARY in subdir2.stem:
                    # If summary
                    _cols = [INDIVIDUAL, MEASURE, AGG]
                    combined_csv_df = combined_csv_df.set_index([EXPERIMENT, *_cols])[
                        VALUE
                    ].unstack(_cols)
                # Prepare in specific format
                csv_fp = subdir1 / f"all_{subdir2.stem}.csv"
                combined_csv_df.to_csv(csv_fp)
