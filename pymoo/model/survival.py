from abc import abstractmethod

import numpy as np

from pymoo.model.population import Population
from pymoo.util.clearing import select_by_clearing, func_select_from_sorted
from pymoo.util.misc import norm_eucl_dist


# ---------------------------------------------------------------------------------------------------------
# Constraint Handling
# ---------------------------------------------------------------------------------------------------------


class LeastInfeasibleSurvival:

    def __init__(self, clearing=True, clearing_delta=0.05) -> None:
        super().__init__()
        self.clearing = clearing
        self.clearing_delta = clearing_delta

    def do(self, problem, pop, n_survive, **kwargs):

        # sort them by cv
        sorted_by_cv = pop.get("CV")[:, 0].argsort()
        pop = pop[sorted_by_cv]

        if self.clearing:

            # calculate the distance from each individual to another - pre-processing for the clearing
            X = pop.get("X")
            D = norm_eucl_dist(problem, X, X)

            # select by clearing using the order defined before
            I = select_by_clearing(pop, D, n_survive, func_select_from_sorted, delta=self.clearing_delta)

            return pop[I]

        else:
            return pop[:n_survive]


# ---------------------------------------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------------------------------------


class Survival:

    def __init__(self,
                 filter_infeasible=True,
                 infeas_survival=LeastInfeasibleSurvival(clearing=False)):
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
        self.infeas_survival = infeas_survival

    def do(self,
           problem,
           pop,
           n_survive,
           n_min_infeas_survive=0,
           return_indices=False,
           **kwargs):

        # make sure the population has at least one individual
        if len(pop) == 0:
            return pop

        n_survive = min(n_survive, len(pop))

        # if the split should be done beforehand
        if self.filter_infeasible and problem.n_constr > 0:

            # initialize the feasible and infeasible population
            feas_pop, infeas_pop = Population(), Population()

            # split feasible and infeasible solutions
            feas, infeas = split_by_feasibility(pop, sort_infeasbible_by_cv=True)

            # if there was no feasible solution was added at all - which means at least one infeasible
            if len(feas) == 0:
                infeas_pop = self.infeas_survival.do(problem, pop[infeas], n_survive, **kwargs)

            # if there are feasible solutions in the population
            else:

                # if feasible solutions do exist
                if len(feas) > 0:
                    feas_pop = self._do(problem, pop[feas], min(len(feas), n_survive), **kwargs)

                # calculate how many individuals are still remaining to be filled up with infeasible ones
                n_remaining = n_survive - len(feas_pop)

                # the maximum of necessary to survive or minimum to be defined being infeasible
                n_infeas_survive = max(n_remaining, n_min_infeas_survive)

                # no more than all infeasible solutions can survive
                n_infeas_survive = min(len(infeas), n_infeas_survive)

                # if infeasible solutions needs to be added
                if n_infeas_survive > 0:
                    infeas_pop = self.infeas_survival.do(problem, pop[infeas], n_infeas_survive, **kwargs)

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
