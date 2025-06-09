"""
Joblib-based parallelization utilities for pymoo.
"""

from . import joblib, requires_joblib


@requires_joblib
def create_joblib_parallelization(n_jobs=-1, backend='threading', **kwargs):
    """
    Create a joblib-based parallelization runner.
    
    Parameters
    ----------
    n_jobs : int, default=-1
        Number of jobs to run in parallel. -1 uses all available processors.
    backend : str, default='threading'
        Parallelization backend ('threading', 'multiprocessing', 'loky').
    **kwargs
        Additional arguments passed to joblib.Parallel.
        
    Returns
    -------
    Parallel
        Configured joblib.Parallel instance.
    """
    return joblib.Parallel(n_jobs=n_jobs, backend=backend, **kwargs)


@requires_joblib
def starmap_joblib(func, iterable, n_jobs=-1, **kwargs):
    """
    Apply function to iterable using joblib parallelization.
    
    Parameters
    ----------
    func : callable
        Function to apply to each element.
    iterable : iterable
        Iterable of arguments to pass to func.
    n_jobs : int, default=-1
        Number of parallel jobs.
    **kwargs
        Additional arguments for joblib.Parallel.
        
    Returns
    -------
    list
        Results from parallel execution.
    """
    parallel = create_joblib_parallelization(n_jobs=n_jobs, **kwargs)
    return parallel(joblib.delayed(func)(args) for args in iterable)