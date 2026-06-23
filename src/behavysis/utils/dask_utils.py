import contextlib
from collections.abc import Generator

from dask.distributed import Client, SpecCluster


@contextlib.contextmanager
def cluster_process(cluster: SpecCluster) -> Generator:
    """Makes a Dask cluster and client.

    Runs the body in the context manager,
    then closes the client and cluster.
    """
    client = Client(cluster)
    print(client.dashboard_link)
    try:
        yield
    finally:
        client.close()
        cluster.close()
