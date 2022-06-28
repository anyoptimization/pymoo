import numpy as np

from pymoo.algorithms.base.local import LocalSearch
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.util.display.single import SingleObjectiveOutput

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize, denormalize


def norm_bounds(pop, problem):
    nxl = normalize(pop.get("xl"), problem.xl, problem.xu)
    nxu = normalize(pop.get("xu"), problem.xl, problem.xu)
    return nxl, nxu


def update_bounds(ind, xl, xu, k, delta):
    _xl = np.copy(xl)
    _xl[k] = ind.X[k] - delta
    ind.set("xl", _xl)

    _xu = np.copy(xu)
    _xu[k] = ind.X[k] + delta
    ind.set("xu", _xu)


class DIRECT(LocalSearch):

    def __init__(self,
                 eps=1e-2,
                 penalty=0.1,
                 n_max_candidates=10,
                 n_max_archive=400,
                 archive_reduct=0.66,
                 output=SingleObjectiveOutput(),
                 **kwargs):
        super().__init__(output=output, **kwargs)
        self.eps = eps
        self.penalty = penalty
        self.n_max_candidates = n_max_candidates
        self.n_max_archive = n_max_archive
        self.archive_reduct = archive_reduct

    def _setup(self, problem, **kwargs):

        xl, xu = problem.bounds()
        X = denormalize(0.5 * np.ones(problem.n_var), xl, xu)

        x0 = Individual(X=X)
        x0.set("xl", xl)
        x0.set("xu", xu)
        x0.set("depth", 0)

        self.x0 = x0

    def _initialize_infill(self, **kwargs):
        return Population.create(self.x0)

    def _potential_optimal(self):
        pop = self.pop

        if len(pop) == 1:
            return pop

        # get the intervals of each individual
        _F, _CV, xl, xu = pop.get("F", "CV", "xl", "xu")
        nF = normalize(_F)
        F = nF + self.penalty * _CV

        # get the length of the interval of each solution
        nxl, nxu = norm_bounds(pop, self.problem)
        length = (nxu - nxl) / 2
        val = length.mean(axis=1)

        # (a) non-dominated set with respect to interval
        obj = np.column_stack([-val, F])

        # an unlimited archive size can cause issues - thus truncate if necessary
        if len(pop) > self.n_max_archive:
            # find the rank of each individual
            _, rank = NonDominatedSorting().do(obj, return_rank=True)

            # calculate the number of solutions after truncation and filter the best ones out
            n_truncated = int(self.archive_reduct * self.n_max_archive)
            I = np.argsort(rank)[:n_truncated]

            # also update all the utility variables defined so far to match the truncation
            pop, F, nxl, nxu, length, val, obj = pop[I], F[I], nxl[I], nxu[I], length[I], val[I], obj[I]
            self.pop = pop

        I = NonDominatedSorting().do(obj, only_non_dominated_front=True)
        candidates, F, xl, xu, val = pop[I], F[I], xl[I], xu[I], val[I]

        # if all candidates are expanded in each iteration this can cause issues - here use crowding distance to decide
        if len(candidates) == 1:
            return candidates
        else:
            if len(candidates) > self.n_max_candidates:
                candidates = RankAndCrowdingSurvival().do(self.problem, pop, n_survive=self.n_max_candidates)

            return candidates

    def _infill(self):

        # the offspring population to finally evaluate and attach to the population
        infills = Population()

        # find the potential optimal solution in the current population
        potential_optimal = self._potential_optimal()

        # for each of those solutions execute the division move
        for current in potential_optimal:

            # find the largest dimension the solution has not been evaluated yet
            nxl, nxu = norm_bounds(current, self.problem)
            k = np.argmax(nxu - nxl)

            # the delta value to be used to get left and right - this is one sixth of the range
            xl, xu = current.get("xl"), current.get("xu")

            delta = (xu[k] - xl[k]) / 6

            # create the left individual
            left_x = np.copy(current.X)
            left_x[k] = xl[k] + delta
            left = Individual(X=left_x)

            # create the right individual
            right_x = np.copy(current.X)
            right_x[k] = xu[k] - delta
            right = Individual(X=right_x)

            # update the boundaries for all the points accordingly
            for ind in [current, left, right]:
                update_bounds(ind, xl, xu, k, delta)

            # create the offspring population, evaluate and attach to current population
            _infill = Population.create(left, right)
            _infill.set("depth", current.get("depth") + 1)

            infills = Population.merge(infills, _infill)

        return infills

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must to be provided."
        self.pop = Population.merge(self.pop, infills)