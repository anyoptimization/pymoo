"""
Dask-based parallelization for pymoo.

This module provides parallelization using Dask distributed computing.
"""


class DaskParallelization:
    """
    Parallelization using Dask distributed client.
    
    This class enables distributed parallel evaluation using Dask,
    which is useful for scaling across multiple machines.
    
    Parameters
    ----------
    client : dask.distributed.Client
        Dask distributed client for submitting tasks
    
    Examples
    --------
    >>> from dask.distributed import Client
    >>> client = Client('scheduler-address:8786')
    >>> runner = DaskParallelization(client)
    >>> # Use with problem.elementwise_runner = runner
    
    Note
    ----
    This requires Dask to be installed separately:
    pip install dask[distributed]
    """

    def __init__(self, client) -> None:
        super().__init__()
        self.client = client

    def __call__(self, f, X):
        """
        Apply function f to each element in X using Dask.
        
        Parameters
        ----------
        f : callable
            Function to apply to each element
        X : list
            List of inputs to process
            
        Returns
        -------
        list
            Results from applying f to each element in X
        """
        jobs = [self.client.submit(f, x) for x in X]
        return [job.result() for job in jobs]

    def __getstate__(self):
        """Handle pickling by removing non-picklable client."""
        state = self.__dict__.copy()
        state.pop("client", None)
        return state