import numpy as np
from pymoo.util.randomized_argsort import randomized_argsort

from pymoo.algorithms.genetic_algorithm import default_is_duplicate
from pymoo.algorithms.nsga3 import get_nadir_point, get_extreme_points_c
from pymoo.experimental.nsgadss import calc_crowding_distance
from pymoo.model.survival import Survival
from pymoo.optimize import minimize
from pymoo.util import plotting
from pymoo.util.misc import cdist
from pymoo.util.non_dominated_sorting import NonDominatedSorting
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymop.factory import get_problem


class ASFSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(True)

        self.extreme_points = None
        self.intercepts = None
        self.nadir_point = None
        self.ideal_point = np.full(ref_dirs.shape[1], np.inf)
        self.worst_point = np.full(ref_dirs.shape[1], -np.inf)

    def _do(self, pop, n_survive, D=None, **kwargs):

        # attributes to be set after the survival
        F = pop.get("F")

        # find or usually update the new ideal point - from feasible solutions
        self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

        # calculate the fronts of the population
        fronts, rank = NonDominatedSorting().do(F, return_rank=True, n_stop_if_ranked=n_survive)
        non_dominated, last_front = fronts[0], fronts[-1]

        # find the extreme points for normalization
        self.extreme_points = get_extreme_points_c(F[non_dominated, :], self.ideal_point,
                                                   extreme_points=self.extreme_points)

        # find the intercepts for normalization and do backup if gaussian elimination fails
        worst_of_population = np.max(F, axis=0)
        worst_of_front = np.max(F[non_dominated, :], axis=0)

        self.nadir_point = get_nadir_point(self.extreme_points, self.ideal_point, self.worst_point,
                                           worst_of_population, worst_of_front)

        #  consider only the population until we come to the splitting front
        I = np.concatenate(fronts)
        pop, rank, F = pop[I], rank[I], F[I]

        # update the front indices for the current population
        counter = 0
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                fronts[i][j] = counter
                counter += 1

        # normalize the whole population by the estimations made
        N = normalize(F, self.ideal_point, self.nadir_point)

        # the final indices of surviving individuals
        survivors = []

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_asf_crowding_distance(N[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]


def calc_asf_crowding_distance(F):
    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        return np.full(n_points, np.inf)
    else:

        crowding = np.zeros((n_points, n_obj))

        for k in range(n_obj):

            I = np.argsort(F[:, k])

            for j in range(n_points):

                current = I[j]

                if j == 0:
                    index = I[j + 1]
                elif j == n_points - 1:
                    index = I[j - 1]
                else:
                    last, next = I[j - 1], I[j + 1]

                    # get the index with largest difference
                    if F[current, k] - F[last, k] > F[next, k] - F[current, k]:
                        index = last
                    else:
                        index = next

                crowding[current, k] = np.abs(F[current, k] - F[index, k])

        crowding = np.min(crowding, axis=1)

        # always keep all extreme points
        asf = np.eye(F.shape[1])
        asf[asf == 0] = 1e6
        F_asf = np.max(F * asf[:, None, :], axis=2).T / 1e6
        for k in range(n_obj):
            crowding[np.argmin(F_asf[:, k])] = np.inf

        return crowding


def normalize(F, ideal_point, nadir_point, utopian_epsilon=0.0):
    utopian_point = ideal_point - utopian_epsilon
    N = (F - utopian_point) / (nadir_point - utopian_point)
    return N


problem = get_problem("dtlz2", n_var=None, n_obj=3, k=5)

n_gen = 400
pop_size = 91
ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=12, scaling=1.0).do()

# create the pareto front for the given reference lines
pf = problem.pareto_front(ref_dirs)

res = minimize(problem,
               method='nsga2',
               method_args={
                   'pop_size': 100,
                   'survival': ASFSurvival()
               },
               termination=('n_gen', n_gen),
               pf=pf,
               save_history=True,
               seed=31,
               disp=True)

plotting.plot(pf, res.F, labels=["Pareto-front", "F"])
