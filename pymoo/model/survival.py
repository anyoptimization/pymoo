from abc import abstractmethod

import numpy as np


class Survival:
    """
    The survival process is implemented inheriting from this class, which selects from a population only
    specific individuals to survive.
    """

    def __init__(self, filter_infeasible) -> None:
        super().__init__()
        self.filter_infeasible = filter_infeasible

    def do(self, problem, pop, n_survive, **kwargs):

        # if the split should be done beforehand
        if self.filter_infeasible and problem.n_constr > 0:
            feasible, infeasible = split_by_feasibility(pop, sort_infeasbible_by_cv=True)

            # finally the survivors to be returned
            survivors = pop.new()

            # if feasible solution do exist
            if len(feasible) > 0:
                survivors = self._do(pop[feasible], min(len(feasible), n_survive), **kwargs)

            # if infeasible solutions needs to be added
            if len(survivors) < n_survive:
                least_infeasible = infeasible[:n_survive - len(feasible)]
                survivors = survivors.merge(pop[least_infeasible])

        else:
            survivors = self._do(pop, n_survive, **kwargs)

        return survivors

    @abstractmethod
    def _do(self, problem, pop, n_survive, **kwargs):
        pass


def split_by_feasibility(pop, sort_infeasbible_by_cv=True):

    CV = pop.get("CV")
    b = (CV <= 0)

    feasible = np.where(b)[0]
    infeasible = np.where(np.logical_not(b))[0]

    if sort_infeasbible_by_cv:
        infeasible = infeasible[np.argsort(CV[infeasible,0])]

    return feasible, infeasible
