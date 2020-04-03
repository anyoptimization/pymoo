import numpy as np

from pymoo.model.repair import Repair
from pymoo.util.misc import at_least_2d_array


def repair_out_of_bounds_manually(X, xl, xu):

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


def repair_out_of_bounds(problem, X):
    return repair_out_of_bounds_manually(X, problem.xl, problem.xu)


class OutOfBoundsRepair(Repair):

    def _do(self, problem, pop, **kwargs):
        X = pop.get("X")
        pop.set("X", repair_out_of_bounds(problem, X))
        return pop
