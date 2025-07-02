import abc

import numpy as np

from pymoo.core.population import Population
from pymoo.core.repair import Repair
from pymoo.util import default_random_state


def is_in_bounds(X, xl, xu):
    return np.where(np.all(np.logical_and(X >= xl, X <= xu), axis=1))[0]


def is_in_bounds_by_problem(problem, X):
    return is_in_bounds(X, problem.xl, problem.xu)


def is_out_of_bounds(X, xl, xu):
    return np.where(np.any(np.logical_or(X < xl, X > xu), axis=1))[0]


def is_out_of_bounds_by_problem(problem, X):
    return is_out_of_bounds(X, problem.xl, problem.xu)


def repeat_bounds(xl, xu, n):
    XL = np.tile(xl, (n, 1))
    XU = np.tile(xu, (n, 1))
    return XL, XU


def repair_clamp(Xp, xl, xu):
    XL, XU = repeat_bounds(xl, xu, len(Xp))

    I = np.where(Xp < XL)
    Xp[I] = XL[I]

    I = np.where(Xp > XU)
    Xp[I] = XU[I]

    return Xp


def repair_periodic(Xp, xl, xu):
    XL, XU = repeat_bounds(xl, xu, len(Xp))

    S = (XU - XL)

    I = np.where(Xp < XL)
    Xp[I] = XU[I] - (XL[I] - Xp[I]) % S[I]

    I = np.where(Xp > XU)
    Xp[I] = XL[I] + (Xp[I] - XU[I]) % S[I]

    return Xp


@default_random_state
def repair_random_init(Xp, X, xl, xu, random_state=None):
    XL, XU = repeat_bounds(xl, xu, len(Xp))

    i, j = np.where(Xp < XL)
    if len(i) > 0:
        Xp[i, j] = XL[i, j] + random_state.random(len(i)) * (X[i, j] - XL[i, j])

    i, j = np.where(Xp > XU)
    if len(i) > 0:
        Xp[i, j] = XU[i, j] - random_state.random(len(i)) * (XU[i, j] - X[i, j])

    return Xp


class BoundsRepair(Repair):

    def _do(self,
            problem,
            pop_or_X,
            check_out_of_bounds=True,
            **kwargs):

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
    def repair_out_of_bounds(self, problem, X, **kwargs):
        pass
