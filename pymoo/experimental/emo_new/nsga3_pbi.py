import numpy as np

from pymoo.cython.function_loader import load_function
from pymoo.algorithms.nsga3 import get_nadir_point, calc_niche_count, niching, get_extreme_points_c
from pymoo.model.survival import Survival
from pymoo.util.non_dominated_sorting import NonDominatedSorting


class ReferenceDirectionSurvivalPBI(Survival):
    def __init__(self, ref_dirs):
        super().__init__(True)
        self.ref_dirs = ref_dirs
        self.extreme_points = None
        self.intercepts = None
        self.nadir_point = None
        self.ideal_point = np.full(ref_dirs.shape[1], np.inf)
        self.worst_point = np.full(ref_dirs.shape[1], -np.inf)

    def get_ref_dirs(self):
        return self.ref_dirs

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
        last_front = fronts[-1]

        # get the reference direction for survival
        ref_dirs = self.get_ref_dirs()

        # associate individuals to niches
        niche_of_individuals, dist_to_niche = associate_to_niches(F, ref_dirs, self.ideal_point, self.nadir_point)
        pop.set('rank', rank, 'niche', niche_of_individuals, 'dist_to_niche', dist_to_niche)

        # if we need to select individuals to survive
        if len(pop) > n_survive:

            # if there is only one front
            if len(fronts) == 1:
                n_remaining = n_survive
                until_last_front = np.array([], dtype=np.int)
                niche_count = np.zeros(len(ref_dirs), dtype=np.int)

            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1])
                niche_count = calc_niche_count(len(ref_dirs), niche_of_individuals[until_last_front])
                n_remaining = n_survive - len(until_last_front)

            S = niching(F[last_front, :], n_remaining, niche_count, niche_of_individuals[last_front],
                        dist_to_niche[last_front])

            survivors = np.concatenate((until_last_front, last_front[S].tolist()))
            pop = pop[survivors]

        return pop


def associate_to_niches(F, niches, ideal_point, nadir_point, utopian_epsilon=0.0):
    utopian_point = ideal_point - utopian_epsilon

    # normalize by ideal point and intercepts
    N = (F - utopian_point) / (nadir_point - utopian_point)

    u = np.tile(niches, (len(N), 1))
    v = np.repeat(N, len(niches), axis=0)

    dist_matrix = load_function("decomposition", "pbi")(u, v, ideal_point, 2)
    dist_matrix = np.reshape(dist_matrix, (len(N), len(niches)))

    niche_of_individuals = np.argmin(dist_matrix, axis=1)
    dist_to_niche = dist_matrix[np.arange(F.shape[0]), niche_of_individuals]

    return niche_of_individuals, dist_to_niche
