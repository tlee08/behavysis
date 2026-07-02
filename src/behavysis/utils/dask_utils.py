import contextlib
from collections.abc import Generator

from dask.distributed import Client, SpecCluster
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
