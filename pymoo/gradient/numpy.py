"""
Numpy gradient toolbox - acts as a fallback when no automatic differentiation library is used.
This module provides the same interface as autograd/jax but raises appropriate errors.
"""
import numpy as np

def value_and_grad(*args, **kwargs):
    raise NotImplementedError("Numpy gradient toolbox does not support automatic differentiation. "
                              "Please use pymoo.gradient.activate('autograd.numpy') or "
                              "pymoo.gradient.activate('jax.numpy') for automatic differentiation.")

def log(*args, **kwargs):
    return np.log(*args, **kwargs)

def sqrt(*args, **kwargs):
    return np.sqrt(*args, **kwargs)

def row_stack(*args, **kwargs):
    return np.vstack(*args, **kwargs)

def triu_indices(*args, **kwargs):
    return np.triu_indices(*args, **kwargs)