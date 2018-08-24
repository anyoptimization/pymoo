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


def split_by_feasibility(pop, sort_infeasbible_by_cv=True):
    # if no constraint violation is provided
    if pop.CV is None:
        return np.arange(pop.size()), np.array([])

    feasible, infeasible = [], []

    for i in range(pop.size()):
        if pop.CV[i, 0] <= 0:
            feasible.append(i)
        else:
            infeasible.append(i)

    if sort_infeasbible_by_cv:
        infeasible = sorted(infeasible, key=lambda i: pop.CV[i,:])

    return np.array(feasible), np.array(infeasible)
