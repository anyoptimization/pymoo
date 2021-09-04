import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.soo.nonconvex.de import DES
from pymoo.docs import parse_doc_string
from pymoo.core.survival import Survival
from pymoo.operators.crossover.dex import DEX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.misc import cdist
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
from pymoo.visualization.video.two_var_one_obj import TwoVariablesOneObjectiveVisualization


# =========================================================================================================
# Survival
# =========================================================================================================


class NichingSurvival(Survival):

    def __init__(self, d1, d2, max_niches, max_crowding) -> None:
        super().__init__(True)
        self.d1 = d1
        self.d2 = d2
        self.max_niches = max_niches
        self.max_crowding = max_crowding

    def _do(self, problem, pop, n_survive=None, out=None, **kwargs):
        F = pop.get("F")
        assert F.shape[1] == 1, "This survival only works for single-objective problems"

        I = np.argsort(F[:, 0])
        pop = pop[I]
        X, F = pop.get("X", "F")

        xl, xu = problem.bounds()
        X = (X - xl) / (xu - xl)

        func_dist = lambda _x, _X: cdist(_x[None, :], _X)[0] / (problem.n_var ** 0.5)

        niches = np.full(len(pop), 0)
        crowding = np.full(len(pop), 0)
        penalty = np.full(len(pop), 0)

        leaders = [0]

        for i in range(1, len(pop)):

            # calculate the distance to all solutions with a better function value than i
            d = func_dist(X[i], X[:i])

            closer_than_d1 = d <= self.d1

            # if the distance is less than d1, then we inherit the niche
            if np.any(closer_than_d1):

                # find the first solution with distance less than d1 - lowest objective value
                J = np.argmax(closer_than_d1)
                niche = niches[J]

            else:

                # create a new niche if maximum has not reached yet
                if len(leaders) < self.max_niches:
                    niche = len(leaders)
                    leaders.append(i)

                # else assign the closest niche
                else:
                    niche = np.argmin(func_dist(X[i], X[leaders]))

            niches[i] = niche

            closer_than_d2 = d <= self.d2

            # if any other solution is in the crowding radius
            if np.any(closer_than_d2):

                # assign the crowding counter for all of them
                J, = np.where(closer_than_d2)
                crowding[J] += 1

                # the penalty is defined by the crowding counter minus the allowed max. allowed crowding
                penalty[i] = max(0, (crowding[J]).max() - self.max_crowding)

        # create the niche assignment matrix A
        A = [[] for _ in range(niches.max() + 1)]
        for i, niche in enumerate(niches):
            A[niche].append(i)

        # set the local optimum rank for each niche
        rank = np.full(len(pop), -1)
        for I in A:
            rank[I] = np.arange(len(I))

        J = np.lexsort((rank, niches, penalty))

        return pop[J[:n_survive]]


# ====================

class LOR2(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=DES("rand"),
                 crossover=DEX(CR=0.7),
                 mutation=PM(eta=20),
                 survival=NichingSurvival(0.1, 0.01, 5, 4),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 display=SingleObjectiveDisplay(),
                 **kwargs):

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         display=display,
                         **kwargs)

        self.default_termination = SingleObjectiveDefaultTermination()

    def _set_optimum(self):
        super()._set_optimum()
        # print(self.opt.get("X"))


parse_doc_string(LOR2.__init__)

from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("himmelblau")

algorithm = LOR2(
    pop_size=100,
    eliminate_duplicates=True,
    callback=TwoVariablesOneObjectiveVisualization()
)

res = minimize(problem,
               algorithm,
               ("n_gen", 10000),
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
