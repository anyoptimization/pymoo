"""Bounce-back out-of-bounds repair strategy."""

import numpy as np

from pymoo.operators.repair.bounds_repair import BoundsRepair
from pymoo.util.misc import at_least_2d_array


def bounce_back(X, xl, xu):
    """Repair out-of-bounds variables by bouncing them back within bounds.

    Args:
        X: Variables to repair.
        xl: Lower bounds.
        xu: Upper bounds.

    Returns:
        Repaired variables.
    """
    only_1d = X.ndim == 1
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
    """Repair out-of-bounds variables using problem bounds.

    Args:
        problem: The optimization problem.
        X: Variables to repair.

    Returns:
        Repaired variables.
    """
    return bounce_back(X, problem.xl, problem.xu)


class BounceBackOutOfBoundsRepair(BoundsRepair):
    """Bounce-back repair operator for out-of-bounds variables."""

    def repair_out_of_bounds(self, problem, X, **kwargs):  # noqa: D417
        """Repair out-of-bounds variables using bounce-back.

        Args:
            problem: The optimization problem.
            X: Population variables.

        Returns:
            Repaired population.
        """
        return bounce_back_by_problem(problem, X)
