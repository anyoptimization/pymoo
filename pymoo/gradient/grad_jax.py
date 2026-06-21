from functools import partial

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from jax import vjp, vmap
from jax._src.api import _jacrev_unravel, _std_basis
from jax.tree_util import tree_map

import pymoo.gradient.toolbox as anp


def jax_elementwise_value_and_grad(f, x):
    out, pullback = vjp(f, x)
    u = _std_basis(out)
    (jac,) = vmap(pullback, in_axes=0)(u)

    grad = tree_map(partial(_jacrev_unravel, out), x, jac)

    return out, grad


def jax_vectorized_value_and_grad(f, x):
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

    grad = {k: np.swapaxes(jac[I], 0, 1) for k, I in cols.items()}

    return out, grad
