"""
Parallelization utilities for pymoo.
"""

from .starmap import StarmapParallelization
from .dask import DaskParallelization
from .joblib import JoblibParallelization
from .ray import RayParallelization

__all__ = [
    'StarmapParallelization', 
    'DaskParallelization', 
    'JoblibParallelization', 
    'RayParallelization'
]