import contextlib
from typing import Callable

from dask.distributed import Client, SpecCluster

from behavysis_pipeline.utils.logging_utils import init_logger

logger = init_logger(__name__)


def cluster_proc_dec(cluster_factory: Callable[[], SpecCluster]):
    """
    `cluster_factory` is a function that returns a Dask cluster.
    This function makes the Dask cluster and client, runs the function,
    then closes the client and cluster.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            cluster = cluster_factory()
            client = Client(cluster)
            logger.debug(client.dashboard_link)
            res = func(*args, **kwargs)
            client.close()
            cluster.close()
            return res

        return wrapper

    return decorator


@contextlib.contextmanager
def cluster_proc_contxt(cluster: SpecCluster):
    """
    Makes a Dask cluster and client, runs the body in the context manager,
    then closes the client and cluster.
    """
    client = Client(cluster)
    logger.debug(client.dashboard_link)
    try:
        yield
    finally:
        client.close()
        cluster.close()
