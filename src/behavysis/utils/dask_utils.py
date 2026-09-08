import contextlib
from collections.abc import Generator

from dask.distributed import Client, Nanny, SpecCluster
from loguru import logger


@contextlib.contextmanager
def cluster_process(cluster: SpecCluster) -> Generator:
    """Makes a Dask cluster and client.

    Runs the body in the context manager,
    then closes the client and cluster.
    """
    client = Client(cluster)
    logger.info(client.dashboard_link)
    try:
        yield
    finally:
        client.close()
        cluster.close()


def gpu_cluster(gpu_ls: list[int | None]) -> SpecCluster:
    """One worker per GPU, each pinned to its own GPU via CUDA_VISIBLE_DEVICES."""
    workers = {}
    for i, gpu in enumerate(gpu_ls):
        options = {"nthreads": 1}
        if gpu is not None:
            options["env"] = {"CUDA_VISIBLE_DEVICES": str(gpu)}
        workers[str(i)] = {"cls": Nanny, "options": options}
    return SpecCluster(workers=workers)
