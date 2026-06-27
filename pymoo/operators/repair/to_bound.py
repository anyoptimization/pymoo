"""Set-to-bounds out-of-bounds repair strategy."""

import numpy as np

from pymoo.operators.repair.bounds_repair import BoundsRepair
from pymoo.util.misc import at_least_2d_array


def set_to_bounds_if_outside(X, xl, xu):
    """Set out-of-bounds variables to the nearest bound.

    Args:
        X: Variables to repair.
        xl: Lower bounds.
        xu: Upper bounds.

    Returns:
        Repaired variables.
    """
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
    """Set out-of-bounds variables to nearest bound using problem bounds.

    Args:
        problem: The optimization problem.
        X: Variables to repair.

    Returns:
        Repaired variables.
    """
    return set_to_bounds_if_outside(X, problem.xl, problem.xu)


class ToBoundOutOfBoundsRepair(BoundsRepair):
    """Set-to-bounds repair operator for out-of-bounds variables."""

    def repair_out_of_bounds(self, problem, X, **kwargs):  # noqa: D417
        """Repair out-of-bounds variables by setting them to bounds.

        Args:
            problem: The optimization problem.
            X: Population variables.

        Returns:
            Repaired population.
        """
        return set_to_bounds_if_outside_by_problem(problem, X)
