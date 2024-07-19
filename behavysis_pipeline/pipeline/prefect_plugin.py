"""
_summary_
"""

from typing import Any

from prefect import flow

from behavysis_pipeline.pipeline import Project


@flow
def process_scaffold(proj: Project, method, *args: Any, **kwargs: Any) -> None:
    """
    Runs the given method on all experiments in the project.
    TODO
    """
    pass
    # # Choosing whether to run the scaffold function in single or multi-processing mode
    # if self.nprocs == 1:
    #     scaffold_func = self._process_scaffold_sp
    # else:
    #     scaffold_func = self._process_scaffold_mp
    # # Running the scaffold function
    # # Starting
    # logging.info("Running %s", method.__name__)
    # # Running
    # dd_ls = scaffold_func(method, *args, **kwargs)
    # if len(dd_ls) > 0:
    #     # Processing all experiments
    #     df = (
    #         pd.DataFrame(dd_ls)
    #         .set_index("experiment")
    #         .sort_index(key=natsort_keygen())
    #     )
    #     # Updating the diagnostics file at each step
    #     self.save_diagnostics(method.__name__, df)
    #     # Finishing
    #     logging.info("Finished %s!\n%s\n%s\n", method.__name__, STR_DIV, STR_DIV)
