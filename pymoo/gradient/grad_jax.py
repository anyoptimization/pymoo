from functools import partial

from jax.config import config

config.update("jax_enable_x64", True)

import pymoo.gradient.toolbox as anp
import numpy as np
from jax import vjp
from jax import vmap
from jax._src.api import _jacrev_unravel, _std_basis
from jax.tree_util import (tree_map)


def jax_elementwise_value_and_grad(f, x):
    out, pullback = vjp(f, x)
    u = _std_basis(out)
    jac, = vmap(pullback, in_axes=0)(u)

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
        for i in range(m):
            cols[k].append(cnt)
            a[cnt, :, i] = 1.0
            cnt += 1

        u[k] = anp.array(a)

    jac, = vmap(pullback, in_axes=0)(u)
    jac = np.array(jac)

    grad = {k: np.swapaxes(jac[I], 0, 1) for k, I in cols.items()}

    return out, grad
