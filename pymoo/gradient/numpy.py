"""Numpy gradient toolbox fallback for when no automatic differentiation is active.

This module provides the same interface as autograd/jax but raises appropriate
errors when automatic differentiation is requested.
"""

import numpy as np


def value_and_grad(*args, **kwargs):
    """Raise error: automatic differentiation not available.

    Raises:
        NotImplementedError: Numpy does not support automatic differentiation.
    """
    raise NotImplementedError(
        "Numpy gradient toolbox does not support automatic differentiation. "
        "Please use pymoo.gradient.activate('autograd.numpy') or "
        "pymoo.gradient.activate('jax.numpy') for automatic differentiation."
    )


def log(*args, **kwargs):
    """Natural logarithm (numpy fallback).

    Args:
        *args: Arguments to numpy.log.
        **kwargs: Keyword arguments to numpy.log.

    Returns:
        Logarithm values.
    """
    return np.log(*args, **kwargs)


def sqrt(*args, **kwargs):
    """Square root (numpy fallback).

    Args:
        *args: Arguments to numpy.sqrt.
        **kwargs: Keyword arguments to numpy.sqrt.

    Returns:
        Square root values.
    """
    return np.sqrt(*args, **kwargs)


def row_stack(*args, **kwargs):
    """Stack arrays vertically (numpy fallback).

    Args:
        *args: Arrays to stack.
        **kwargs: Keyword arguments to numpy.vstack.

    Returns:
        Stacked array.
    """
    return np.vstack(*args, **kwargs)


def triu_indices(*args, **kwargs):
    """Return indices of upper triangular matrix (numpy fallback).

    Args:
        *args: Arguments to numpy.triu_indices.
        **kwargs: Keyword arguments to numpy.triu_indices.

    Returns:
        Indices of upper triangular matrix.
    """
    return np.triu_indices(*args, **kwargs)
