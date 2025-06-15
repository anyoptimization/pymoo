"""
Parallelization utilities for pymoo.

This module provides optional parallelization capabilities using external libraries.
"""

# Try to import parallelization dependencies
try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False
    joblib = None

try:
    import ray
    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False
    ray = None


class _ParallelizationNotAvailable:
    """Helper class to provide informative error messages when parallelization libraries are not available."""
    
    def __init__(self, library_name):
        self.library_name = library_name
    
    def __getattr__(self, name):
        raise ImportError(
            f"Parallelization feature '{name}' requires {self.library_name}.\n"
            "Install with one of:\n"
            f"  pip install pymoo[parallelization]\n"
            f"  pip install pymoo[full]\n"
            f"  pip install {self.library_name}"
        )


# Export parallelization tools with fallback error handling
if not _JOBLIB_AVAILABLE:
    joblib = _ParallelizationNotAvailable("joblib")

if not _RAY_AVAILABLE:
    ray = _ParallelizationNotAvailable("ray")


# Import parallelization classes
from .starmap import StarmapParallelization
from .dask import DaskParallelization
from .joblib import JoblibParallelization
from .ray import RayParallelization

# Export all parallelization functionality
__all__ = [
    'joblib', 'ray', 'is_joblib_available', 'is_ray_available',
    'StarmapParallelization', 'DaskParallelization', 
    'JoblibParallelization', 'RayParallelization'
]


def is_joblib_available():
    """Check if joblib is available for parallelization."""
    return _JOBLIB_AVAILABLE


def is_ray_available():
    """Check if ray is available for parallelization."""
    return _RAY_AVAILABLE




