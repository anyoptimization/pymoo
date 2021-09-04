from abc import abstractmethod

import numpy as np

from pymoo.core.population import Population


# ---------------------------------------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------------------------------------


class Survival:

    def __init__(self, filter_infeasible=True):
        super().__init__()
        self.filter_infeasible = filter_infeasible

    def do(self,
           problem,
           pop,
           *args,
           n_survive=None,
           return_indices=False,
           **kwargs):

        # make sure the population has at least one individual
        if len(pop) == 0:
            return pop

        if n_survive is None:
            n_survive = len(pop)

        n_survive = min(n_survive, len(pop))

        # if the split should be done beforehand
        if self.filter_infeasible and problem.n_constr > 0:

            # split feasible and infeasible solutions
            feas, infeas = split_by_feasibility(pop, eps=0.0, sort_infeasbible_by_cv=True)

            if len(feas) == 0:
                survivors = Population()
            else:
                survivors = self._do(problem, pop[feas], *args, n_survive=min(len(feas), n_survive), **kwargs)

            # calculate how many individuals are still remaining to be filled up with infeasible ones
            n_remaining = n_survive - len(survivors)

            # if infeasible solutions needs to be added
            if n_remaining > 0:
                survivors = Population.merge(survivors, pop[infeas[:n_remaining]])

        else:
            survivors = self._do(problem, pop, *args, n_survive=n_survive, **kwargs)

        if return_indices:
            H = {}
            for k, ind in enumerate(pop):
                H[ind] = k
            return [H[survivor] for survivor in survivors]
        else:
            return survivors

    @abstractmethod
    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        pass


def split_by_feasibility(pop, eps=0.0, sort_infeasbible_by_cv=True):
    CV = pop.get("CV")

    b = (CV <= eps)

    feasible = np.where(b)[0]
    infeasible = np.where(~b)[0]

    if sort_infeasbible_by_cv:
        infeasible = infeasible[np.argsort(CV[infeasible, 0])]

    return feasible, infeasible


def calc_adapt_eps(pop):
    cv = pop.get("CV")[:, 0]
    cv_mean = np.median(cv)
    fr = (cv <= 0).sum() / len(cv)
    return cv_mean * fr
