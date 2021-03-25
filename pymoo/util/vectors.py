import numpy as np


def max_alpha(point, direction, xl, xu, mode="one_hits_bound"):
    bounds = []

    if xl is not None:
        bounds.append(xl)

    if xu is not None:
        bounds.append(xu)

    if len(bounds) == 0:
        return np.inf

    # the bounds in one array
    bounds = np.column_stack(bounds)

    # if the direction is too small we can not divide by 0 - nan will make it being ignored
    dir = direction.copy()
    dir[dir == 0] = np.nan

    # calculate the max factor to be not out of bounds
    val = (bounds - point[:, None]) / dir[:, None]

    # remove nan values by setting them to a negative number
    val[np.isnan(val)] = - np.inf

    # if no value there - no bound exist
    if len(val) == 0:
        return np.inf
    # otherwise return the minimum of values considered
    else:
        if mode == "one_hits_bound":
            if not np.any(val >= 0):
                return 0.0
            else:
                return val[val >= 0].min()
        else:
            return val.max()
