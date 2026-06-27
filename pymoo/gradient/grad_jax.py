"""JAX-based automatic differentiation backend."""

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
from jax import vjp, vmap  # noqa: E402
from jax._src.api import _jacrev_unravel, _std_basis  # noqa: E402
from jax.tree_util import tree_map  # noqa: E402

import pymoo.gradient.toolbox as anp  # noqa: E402


def jax_elementwise_value_and_grad(f, x):
    """Compute value and gradient for elementwise function using JAX.

    Args:
        f: Function to differentiate.
        x: Input value.

    Returns:
        Tuple of (output, gradient).
    """
    out, pullback = vjp(f, x)
    u = _std_basis(out)
    (jac,) = vmap(pullback, in_axes=0)(u)

    grad = tree_map(partial(_jacrev_unravel, out), x, jac)

    return out, grad


def jax_vectorized_value_and_grad(f, x):
    """Compute value and gradient for vectorized function using JAX.

    Args:
        f: Function to differentiate.
        x: Input array of shape (n, p).

    Returns:
        Tuple of (output, gradient).
    """
    out, pullback = vjp(f, x)

    ncols = sum([v.shape[1] for v in out.values()])

    u = dict()
    cols = dict()
    cnt = 0
    for k, v in out.items():
        if k not in cols:
            cols[k] = []

        n, m = v.shape
        a = np.zeros((ncols, n, m))
        cols[k].extend(range(cnt, cnt + m))
        a[cnt : cnt + m, :, :] = np.eye(m)[:, np.newaxis, :]
        cnt += m

        u[k] = anp.array(a)

    (jac,) = vmap(pullback, in_axes=0)(u)
    jac = np.array(jac)

    grad = {k: np.swapaxes(jac[I], 0, 1) for k, I in cols.items()}  # noqa: E741

    return out, grad
