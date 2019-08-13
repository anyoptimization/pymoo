import numpy as np

from pymoo.model.repair import Repair
from pymoo.util.misc import at_least_2d_array


def bounds_back(problem, X):
    only_1d = (X.ndim == 1)
    X = at_least_2d_array(X)

    if problem.xl is not None and problem.xu is not None:
        xl = np.repeat(problem.xl[None, :], X.shape[0], axis=0)
        xu = np.repeat(problem.xu[None, :], X.shape[0], axis=0)

        # otherwise bounds back into the feasible space
        _range = xu - xl
        X[X < xl] = (xl + np.mod((xl - X), _range))[X < xl]
        X[X > xu] = (xu - np.mod((X - xu), _range))[X > xu]

    if only_1d:
        return X[0, :]
    else:
        return X


class BoundsBackRepair(Repair):

    def _do(self, problem, pop, **kwargs):
        # bring back to bounds if violated through crossover - bounce back strategy
        X = pop.get("X")
        xl = np.repeat(problem.xl[None, :], X.shape[0], axis=0)
        xu = np.repeat(problem.xu[None, :], X.shape[0], axis=0)

        # otherwise bounds back into the feasible space
        _range = xu - xl
        X[X < xl] = (xl + np.mod((xl - X), _range))[X < xl]
        X[X > xu] = (xu - np.mod((X - xu), _range))[X > xu]

        return pop.new("X", X)
