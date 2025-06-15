"""
Starmap-based parallelization for pymoo.

This module provides a wrapper around arbitrary starmap functions for parallel evaluation.
"""


class StarmapParallelization:
    """
    Parallelization using a custom starmap function.
    
    This is a generic wrapper that can be used with any starmap-like function,
    such as multiprocessing.Pool.starmap or concurrent.futures.ThreadPoolExecutor.map.
    
    Parameters
    ----------
    starmap : callable
        A starmap function that applies a function to an iterable of arguments.
        Should have signature: starmap(func, iterable_of_args)
    
    Examples
    --------
    >>> from multiprocessing.pool import ThreadPool
    >>> pool = ThreadPool(4)
    >>> runner = StarmapParallelization(pool.starmap)
    >>> # Use with problem.elementwise_runner = runner
    """

    def __init__(self, starmap) -> None:
        super().__init__()
        self.starmap = starmap

    def __call__(self, f, X):
        """
        Apply function f to each element in X using starmap.
        
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
        return list(self.starmap(f, [[x] for x in X]))

    def __getstate__(self):
        """Handle pickling by removing non-picklable starmap function."""
        state = self.__dict__.copy()
        state.pop("starmap", None)
        return state