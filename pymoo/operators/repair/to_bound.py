import numpy as np

from pymoo.operators.repair.bounds_repair import BoundsRepair
from pymoo.util.misc import at_least_2d_array


def set_to_bounds_if_outside(X, xl, xu):
    only_1d = (X.ndim == 1)
    X = at_least_2d_array(X)

    if xl is not None:
        xl = np.repeat(xl[None, :], X.shape[0], axis=0)
        X[X < xl] = xl[X < xl]

    if xu is not None:
        xu = np.repeat(xu[None, :], X.shape[0], axis=0)
        X[X > xu] = xu[X > xu]

    if only_1d:
        return X[0, :]
    else:
        return X


def set_to_bounds_if_outside_by_problem(problem, X):
    return set_to_bounds_if_outside(X, problem.xl, problem.xu)


class ToBoundOutOfBoundsRepair(BoundsRepair):

    def repair_out_of_bounds(self, problem, X):
        return set_to_bounds_if_outside_by_problem(problem, X)
