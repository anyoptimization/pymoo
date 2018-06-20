import numpy as np
from scipy.spatial.distance import cdist

from pymoo.model.survival import Survival
from pymoo.rand import random
from pymoo.util.misc import normalize_by_asf_interceptions
from pymoo.util.non_dominated_rank import NonDominatedRank


class ReferenceLineSurvival(Survival):
    def __init__(self, ref_points, epsilon, weights):
        super().__init__()
        self.ref_points = ref_points
        self.epsilon = epsilon
        self.weights = weights

    def _do(self, pop, n_survive, data, return_only_index=False):

        fronts = NonDominatedRank.calc_as_fronts(pop.F, pop.G)

        # all indices to survive
        survival = []

        for front in fronts:
            if len(survival) + len(front) > n_survive:
                break
            survival.extend(front)

        # filter the front to only relevant entries
        pop.filter(survival + front)
        survival = list(range(0, len(survival)))
        last_front = np.arange(len(survival), pop.size())
        # Indices of last front members that survived
        survived = []

        # N = normalize_by_asf_interceptions(pop.F, return_bounds=False)
        # N = normalize(pop.F, np.zeros(pop.F.shape[1]), np.ones(pop.F.shape[1]))

        # if the last front needs to be splitted
        n_remaining = n_survive - len(survival)


        if n_remaining > 0:

            # dist_matrix = calc_perpendicular_dist_matrix(N, self.ref_dirs)
            # F_bar = map_linear(pop.F, pop.F)
            # Z_bar = map_linear(self.ref_points, pop.F)
            # dist_matrix = calc_ref_dist_matrix(F_bar, Z_bar, self.weights)
            # point_distance_matrix = calc_ref_dist_matrix(F_bar[last_front, :], F_bar[last_front, :], self.weights)

            dist_matrix = calc_ref_dist_matrix(pop.F, self.ref_points, self.weights)
            point_distance_matrix = calc_ref_dist_matrix(pop.F[last_front, :], pop.F[last_front, :], self.weights)

            # point_distance_matrix = cdist(N[last_front, :], N[last_front, :])

            niche_of_individuals = np.argmin(dist_matrix, axis=1)
            min_dist_matrix = dist_matrix[np.arange(len(dist_matrix)), niche_of_individuals]

            # for each reference direction the niche count
            niche_count = np.zeros(len(self.ref_points))
            for i in niche_of_individuals[survival]:
                niche_count[i] += 1

            # relative index now to dist and the niches
            min_dist_matrix = min_dist_matrix[last_front]
            niche_of_individuals = niche_of_individuals[last_front]

            # boolean array of elements that survive if true
            survival_last_front = np.full(len(last_front), False)

            while n_remaining > 0:

                # all niches where new individuals can be assigned to
                next_niches_list = np.unique(niche_of_individuals[np.logical_not(survival_last_front)])

                # pick a niche with minimum assigned individuals - break tie if necessary
                next_niche_count = niche_count[next_niches_list]
                next_niche = np.where(next_niche_count == next_niche_count.min())[0]
                next_niche = next_niche[random.randint(0, len(next_niche))]
                next_niche = next_niches_list[next_niche]

                # indices of individuals in last front to assign niche to
                next_ind = np.where(niche_of_individuals[np.logical_not(survival_last_front)] == next_niche)[0]
                next_ind = np.where(np.logical_not(survival_last_front))[0][next_ind]

                # Pick the closest point
                next_ind = next_ind[np.argmin(min_dist_matrix[next_ind])]

                # Find surrounding points within trust region
                surrounding_points = np.where(point_distance_matrix[next_ind] < self.epsilon)[0]

                # Clear points in trust region
                survival_last_front[surrounding_points] = True

                # Add selected point to survived population
                survived.append(next_ind)

                if np.all(survival_last_front):
                    survival_last_front = np.full(len(last_front), False)
                    survival_last_front[survived] = True

                niche_count[next_niche] += 1
                n_remaining -= 1

        survival.extend(last_front[survived])
        if return_only_index:
            return survival

        # now truncate the population
        pop.filter(survival)

        return pop

def map_linear(xa, xb):
    f_min = xb.min(axis=0)
    f_max = xb.max(axis=0)
    # f_max = np.ones(len(f_min))
    return (xa - f_min)/(f_max-f_min)



def calc_perpendicular_dist(v, u):
    norm_u = np.linalg.norm(u)
    scalar_proj = np.dot(v, u) / norm_u
    proj = scalar_proj * u / norm_u
    return np.linalg.norm(proj - v)


def calc_ref_dist_matrix_slow(F, ref_dirs):
    F_min, F_max = F.min(axis=0), F.max(axis=0)
    w = np.full(F.shape[1], 1 / F.shape[1])
    def calc_dist_matrix(f, r):
        return np.sqrt(np.sum(w*((f-r)/(F_max-F_min))**2))

    return cdist(F, ref_dirs, metric=calc_dist_matrix)

def calc_ref_dist_matrix(F, ref_dirs, weights, F_min=None, F_max=None):
    if F_min is None:
        F_min = F.min(axis=0)
    if F_max is None:
        F_max = F.max(axis=0)
    if weights is None:
        weights = np.full(F.shape[1], 1 / F.shape[1])


    r = np.tile(ref_dirs, (len(F), 1))
    f = np.repeat(F, len(ref_dirs), axis=0)

    if not np.array_equal(F_max, F_min):
        matrix = np.sqrt(np.sum(weights * ((f - r) / (F_max - F_min)) ** 2, axis=1))
    else:
        matrix = np.sqrt(np.sum(weights * (f - r) ** 2, axis=1))

    # matrix = np.sqrt(np.sum(weights * ((f - r) / (F_max - F_min)) ** 2, axis=1))
    # matrix = np.sqrt(np.sum((f - r) ** 2, axis=1))
    matrix = np.reshape(matrix, (F.shape[0], ref_dirs.shape[0]))
    return matrix