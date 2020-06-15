from abc import abstractmethod

import numpy as np

from pymoo.model.population import Population
from pymoo.util.clearing import select_by_clearing
from pymoo.util.misc import norm_eucl_dist


# ---------------------------------------------------------------------------------------------------------
# Default Constraint Handling
# ---------------------------------------------------------------------------------------------------------


class LeastInfeasibleSurvival:

    def do(self, _, pop, n_survive, **kwargs):
        pop = pop[pop.get("CV")[:, 0].argsort()]
        return pop[:n_survive]


class LeastInfeasibleWithClearingSurvival:

    def do(self, problem, pop, n_survive, **kwargs):

        # calculate the distance from each individual to another - pre-processing for the clearing
        X = pop.get("X")
        D = norm_eucl_dist(problem, X, X)

        def func_select_by_constraint_violation(pop):
            CV = pop.get("CV")
            return CV[:, 0].argmin()

        I = select_by_clearing(pop, D, n_survive, func_select_by_constraint_violation)

        return pop[I]


# ---------------------------------------------------------------------------------------------------------
# Generic Survival
# ---------------------------------------------------------------------------------------------------------


class Survival:

    def __init__(self,
                 filter_infeasible=True,
                 cv_survival=LeastInfeasibleSurvival()) -> None:
        """
        The survival process is implemented inheriting from this class, which selects from a population only
        specific individuals to survive. This base class can take care of splitting the feasible and infeasible
        solutions first. By default infeasible solutions are simply sorted by their constraint violation.


        Parameters
        ----------

        filter_infeasible : bool
            Whether for the survival infeasible solutions should be filtered first

        """
        super().__init__()
        self.filter_infeasible = filter_infeasible
        self.cv_survival = cv_survival

    def do(self, problem, pop, n_survive, return_indices=False, **kwargs):

        # make sure the population has at least one individual
        if len(pop) == 0:
            return pop

        # if the split should be done beforehand
        if self.filter_infeasible and problem.n_constr > 0:
            feasible, infeasible = split_by_feasibility(pop, sort_infeasbible_by_cv=True)

            # initialize the feasible and infeasible population
            feas_pop, infeas_pop = Population(), Population()

            # if there was no feasible solution was added at all - which means at least one infeasible
            if len(feasible) == 0:
                infeas_pop = self.cv_survival.do(problem, pop[infeasible], n_survive)

            # if there are feasible solutions in the population
            else:

                # if feasible solution do exist
                if len(feasible) > 0:
                    feas_pop = self._do(problem, pop[feasible], min(len(feasible), n_survive), **kwargs)

                # calculate how many individuals are still remaining to be filled up with infeasible ones
                n_remaining = n_survive - len(feas_pop)

                # if infeasible solutions needs to be added
                if n_remaining > 0:
                    infeas_pop = self.cv_survival.do(problem, pop[infeasible], n_remaining)

            survivors = Population.merge(feas_pop, infeas_pop)

        else:
            survivors = self._do(problem, pop, n_survive, **kwargs)

        if return_indices:
            H = {}
            for k, ind in enumerate(pop):
                H[ind] = k
            return [H[survivor] for survivor in survivors]
        else:
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
        infeasible = infeasible[np.argsort(CV[infeasible, 0])]

    return feasible, infeasible
