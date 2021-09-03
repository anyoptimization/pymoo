import numpy as np

from pymoo.util.misc import list_of_dicts_unique, list_of_dicts_filter


def filter_by(data, filters, return_group=False):
    yield from filter_by_rec(data, filters, {}, return_scope=return_group)


def filter_by_rec(data, filters, scope, return_scope):
    if len(filters) == 0:
        if return_scope:
            yield scope, data
        else:
            yield data
    else:
        entry = filters[0]

        if isinstance(entry, tuple):
            k, vals = entry
        else:
            k = entry
            vals = list_of_dicts_unique(data, k)

        for v in vals:
            _data = list_of_dicts_filter(data, (k, v))
            _filters = filters[1:]
            _scope = {**dict(scope), **{k: v}}
            yield from filter_by_rec(_data, _filters, _scope, return_scope)


def fill_forward_if_nan(vals):
    current = np.nan
    for k in range(len(vals)):

        if not np.isnan(vals[k]):
            current = vals[k]

        if np.isnan(vals[k]):
            vals[k] = current
    return vals


def at_least2d(x, expand="c"):
    if x.ndim == 1:
        if expand == "c":
            return x[:, None]
        elif expand == "r":
            return x[None, :]
    else:
        return x
