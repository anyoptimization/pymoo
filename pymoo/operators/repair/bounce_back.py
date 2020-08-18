import numpy as np

from pymoo.operators.repair.bounds_repair import BoundsRepair
from pymoo.util.misc import at_least_2d_array


def bounce_back(X, xl, xu):
    only_1d = (X.ndim == 1)
    X = at_least_2d_array(X)

    xl = np.repeat(xl[None, :], X.shape[0], axis=0)
    xu = np.repeat(xu[None, :], X.shape[0], axis=0)

    # otherwise bounds back into the feasible space
    _range = xu - xl
    X[X < xl] = (xl + np.mod((xl - X), _range))[X < xl]
    X[X > xu] = (xu - np.mod((X - xu), _range))[X > xu]

    if only_1d:
        return X[0, :]
    else:
        return X


def bounce_back_by_problem(problem, X):
    return bounce_back(X, problem.xl, problem.xu)


class BounceBackOutOfBoundsRepair(BoundsRepair):

    def repair_out_of_bounds(self, problem, X, **kwargs):
        return bounce_back_by_problem(problem, X)
