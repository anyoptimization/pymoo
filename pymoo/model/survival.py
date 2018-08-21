from abc import abstractmethod

import numpy as np


class Survival:
    """
    The survival process is implemented inheriting from this class, which selects from a population only
    specific individuals to survive.
    """

    def __init__(self) -> None:
        super().__init__()

    def do(self, pop, n_survive, **kwargs):
        self._do(pop, n_survive, **kwargs)

    @abstractmethod
    def _do(self, pop, n_survive, **kwargs):
        pass


def split_by_feasibility(pop):
    # if no constraint violation is provided
    if pop.CV is None:
        return np.arange(pop.size()), np.array([])

    P = np.argsort(pop.CV[:, 0])

    # split into a set of feasible and infeasible
    b = pop.CV[P, 0] <= 0
    feasible = P[np.where(b)[0]]
    infeasible = P[np.where(np.logical_not(b))[0]]

    return feasible, infeasible
