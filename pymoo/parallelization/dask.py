"""
Dask-based parallelization for pymoo.
"""


class DaskParallelization:
    """Parallelization using Dask distributed client.
    
    Parameters
    ----------
    client : dask.distributed.Client
        Dask client for distributed computing
    """

    def __init__(self, client):
        self.client = client

    def __call__(self, f, X):
        jobs = [self.client.submit(f, x) for x in X]
        return [job.result() for job in jobs]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("client", None)
        return state