import numpy as np

from pymoo.operators.repair.bounds_repair import BoundsRepair
from pymoo.util.misc import at_least_2d_array


def set_to_bounds_if_outside(X, xl, xu):
    _X, only_1d = at_least_2d_array(X, return_if_reshaped=True)

    if xl is not None:
        xl = np.repeat(xl[None, :], _X.shape[0], axis=0)
        _X[_X < xl] = xl[_X < xl]

    if xu is not None:
        xu = np.repeat(xu[None, :], _X.shape[0], axis=0)
        _X[_X > xu] = xu[_X > xu]

    if only_1d:
        return _X[0, :]
    else:
        return _X


def set_to_bounds_if_outside_by_problem(problem, X):
    return set_to_bounds_if_outside(X, problem.xl, problem.xu)


class ToBoundOutOfBoundsRepair(BoundsRepair):

    def repair_out_of_bounds(self, problem, X, **kwargs):
        return set_to_bounds_if_outside_by_problem(problem, X)
