"""Bounds repair operators and utilities."""

import abc

import numpy as np

from pymoo.core.population import Population
from pymoo.core.repair import Repair
from pymoo.util import default_random_state


def is_in_bounds(X, xl, xu):
    """Check which individuals are within bounds.

    Args:
        X: Population variables.
        xl: Lower bounds.
        xu: Upper bounds.

    Returns:
        Indices of individuals within bounds.
    """
    return np.where(np.all(np.logical_and(X >= xl, X <= xu), axis=1))[0]


def is_in_bounds_by_problem(problem, X):
    """Check which individuals are within problem bounds.

    Args:
        problem: The optimization problem.
        X: Population variables.

    Returns:
        Indices of individuals within bounds.
    """
    return is_in_bounds(X, problem.xl, problem.xu)


def is_out_of_bounds(X, xl, xu):
    """Check which individuals are out of bounds.

    Args:
        X: Population variables.
        xl: Lower bounds.
        xu: Upper bounds.

    Returns:
        Indices of individuals out of bounds.
    """
    return np.where(np.any(np.logical_or(X < xl, X > xu), axis=1))[0]


def is_out_of_bounds_by_problem(problem, X):
    """Check which individuals are out of problem bounds.

    Args:
        problem: The optimization problem.
        X: Population variables.

    Returns:
        Indices of individuals out of bounds.
    """
    return is_out_of_bounds(X, problem.xl, problem.xu)


def repeat_bounds(xl, xu, n):
    """Repeat bounds for n individuals.

    Args:
        xl: Lower bounds.
        xu: Upper bounds.
        n: Number of individuals.

    Returns:
        Repeated lower and upper bounds.
    """
    XL = np.tile(xl, (n, 1))
    XU = np.tile(xu, (n, 1))
    return XL, XU


def repair_clamp(Xp, xl, xu):
    """Repair variables by clamping to bounds.

    Args:
        Xp: Variables to repair.
        xl: Lower bounds.
        xu: Upper bounds.

    Returns:
        Repaired variables.
    """
    XL, XU = repeat_bounds(xl, xu, len(Xp))

    I = np.where(Xp < XL)  # noqa: E741
    Xp[I] = XL[I]

    I = np.where(Xp > XU)  # noqa: E741
    Xp[I] = XU[I]

    return Xp


def repair_periodic(Xp, xl, xu):
    """Repair variables using periodic/wrapping strategy.

    Args:
        Xp: Variables to repair.
        xl: Lower bounds.
        xu: Upper bounds.

    Returns:
        Repaired variables.
    """
    XL, XU = repeat_bounds(xl, xu, len(Xp))

    S = XU - XL

    I = np.where(Xp < XL)  # noqa: E741
    Xp[I] = XU[I] - (XL[I] - Xp[I]) % S[I]

    I = np.where(Xp > XU)  # noqa: E741
    Xp[I] = XL[I] + (Xp[I] - XU[I]) % S[I]

    return Xp


@default_random_state
def repair_random_init(Xp, X, xl, xu, random_state=None):
    """Repair variables by random re-initialization.

    Args:
        Xp: Variables to repair.
        X: Original variables.
        xl: Lower bounds.
        xu: Upper bounds.
        random_state: Random state for reproducibility.

    Returns:
        Repaired variables.
    """
    XL, XU = repeat_bounds(xl, xu, len(Xp))

    i, j = np.where(Xp < XL)
    if len(i) > 0:
        Xp[i, j] = XL[i, j] + random_state.random(len(i)) * (X[i, j] - XL[i, j])

    i, j = np.where(Xp > XU)
    if len(i) > 0:
        Xp[i, j] = XU[i, j] - random_state.random(len(i)) * (XU[i, j] - X[i, j])

    return Xp


class BoundsRepair(Repair):
    """Base class for bounds repair operators."""

    def _do(  # noqa: D417
        self,
        problem,
        pop_or_X,
        check_out_of_bounds=True,
        **kwargs,
    ):
        """Repair out-of-bounds variables.

        Args:
            problem: The optimization problem.
            pop_or_X: Population or variable array to repair.
            check_out_of_bounds: Whether to check bounds after repair.

        Returns:
            Repaired population or variables.
        """
        is_array = not isinstance(pop_or_X, Population)

        X = pop_or_X if is_array else pop_or_X.get("X")

        X = self.repair_out_of_bounds(problem, X, **kwargs)

        assert len(is_out_of_bounds_by_problem(problem, X)) == 0

        if is_array:
            return X
        else:
            pop_or_X.set("X", X)
            return pop_or_X

    @abc.abstractmethod
    def repair_out_of_bounds(self, problem, X, **kwargs):  # noqa: D417
        """Repair out-of-bounds variables.

        Args:
            problem: The optimization problem.
            X: Population variables.

        Returns:
            Repaired variables.
        """
        pass
