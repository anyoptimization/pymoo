from abc import abstractmethod

import numpy as np

from pymoo.core.population import Population
from pymoo.util import default_random_state


# ---------------------------------------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------------------------------------


class Survival:

    def __init__(self, filter_infeasible=True):
        super().__init__()
        self.filter_infeasible = filter_infeasible

    @default_random_state
    def do(self,
           problem,
           pop,
           *args,
           n_survive=None,
           random_state=None,
           return_indices=False,
           **kwargs):

        # make sure the population has at least one individual
        if len(pop) == 0:
            return pop

        if n_survive is None:
            n_survive = len(pop)

        n_survive = min(n_survive, len(pop))

        # if the split should be done beforehand
        if self.filter_infeasible and problem.has_constraints():

            # split feasible and infeasible solutions
            feas, infeas = split_by_feasibility(pop, sort_infeas_by_cv=True)

            if len(feas) == 0:
                survivors = Population()
            else:
                survivors = self._do(problem, pop[feas], *args, n_survive=min(len(feas), n_survive),
                                     random_state=random_state, **kwargs)

            # calculate how many individuals are still remaining to be filled up with infeasible ones
            n_remaining = n_survive - len(survivors)

            # if infeasible solutions needs to be added
            if n_remaining > 0:
                survivors = Population.merge(survivors, pop[infeas[:n_remaining]])

        else:
            survivors = self._do(problem, pop, *args, n_survive=n_survive, random_state=random_state, **kwargs)

        if return_indices:
            H = {}
            for k, ind in enumerate(pop):
                H[ind] = k
            return [H[survivor] for survivor in survivors]
        else:
            return survivors

    @abstractmethod
    def _do(self, problem, pop, *args, n_survive=None, random_state=None, **kwargs):
        pass


class ToReplacement(Survival):

    def __init__(self, survival):
        super().__init__(False)
        self.survival = survival

    def _do(self, problem, pop, off, random_state=None, **kwargs):
        merged = Population.merge(pop, off)
        I = self.survival.do(problem, merged, n_survive=len(merged), return_indices=True, random_state=random_state, **kwargs)
        merged.set("__rank__", I)

        for k in range(len(pop)):
            if off[k].get("__rank__") < pop[k].get("__rank__"):
                pop[k] = off[k]

        return pop


def split_by_feasibility(pop, sort_infeas_by_cv=True, sort_feas_by_obj=False, return_pop=False):
    F, CV, b = pop.get("F", "CV", "FEAS")

    feasible = np.where(b)[0]
    infeasible = np.where(~b)[0]

    if sort_infeas_by_cv:
        infeasible = infeasible[np.argsort(CV[infeasible, 0])]

    if sort_feas_by_obj:
        feasible = feasible[np.argsort(F[feasible, 0])]

    if not return_pop:
        return feasible, infeasible
    else:
        return feasible, infeasible, pop[feasible], pop[infeasible]
