import numpy as np
from scipy.spatial.distance import cdist

from pymoo.model.survival import Survival
from pymoo.rand import random
from pymoo.util.misc import normalize_by_asf_interceptions
from pymoo.util.non_dominated_rank import NonDominatedRank


class ReferenceLineSurvival(Survival):
    def __init__(self, ref_dirs):
        super().__init__()
        self.ref_dirs = ref_dirs

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
        remaining_index = np.arange(0, len(last_front))

        N = normalize_by_asf_interceptions(pop.F, return_bounds=False)
        #N = normalize(pop.F, np.zeros(pop.F.shape[1]), np.ones(pop.F.shape[1]))
        #N = normalize(pop.F, np.zeros(pop.F.shape[1]), np.ones(pop.F.shape[1]))

        dist_matrix = calc_perpendicular_dist_matrix(N, self.ref_dirs)

        # if the last front needs to be splitted
        n_remaining = n_survive - len(survival)
        if n_remaining > 0:

            niche_of_individuals = np.argmin(dist_matrix, axis=1)
            min_dist_matrix = dist_matrix[np.arange(len(dist_matrix)), niche_of_individuals]

            # for each reference direction the niche count
            niche_count = np.zeros(len(self.ref_dirs))
            for i in niche_of_individuals[survival]:
                niche_count[i] += 1

            while n_remaining > 0:

                # get all remaining individuals from the last front
                last_front_remaining = last_front[remaining_index]

                # all niches where new individuals can be assigned to
                next_niches_list = np.unique(niche_of_individuals[last_front_remaining])

                # pick a niche with minimum assigned individuals - break tie if necessary
                next_niche_count = niche_count[next_niches_list]
                next_niche = np.where(next_niche_count == next_niche_count.min())[0]
                next_niche = next_niche[random.randint(0, len(next_niche))]
                next_niche = next_niches_list[next_niche]

                # indices of individuals in last front to assign niche to
                next_ind = np.where(niche_of_individuals[last_front_remaining] == next_niche)[0]

                if len(next_ind) == 1:
                    next_ind = next_ind[0]
                elif niche_count[next_niche] == 0:
                    next_ind = next_ind[np.argmin(min_dist_matrix[next_ind])]
                else:
                    next_ind = next_ind[random.randint(0, len(next_ind))]

                survival.append(last_front[next_ind])
                remaining_index.remove(next_ind)
                niche_count[next_niche] += 1
                n_remaining -= 1

        if return_only_index:
            return survival

        # now truncate the population
        pop.filter(survival)

        return pop


def calc_perpendicular_dist(v, u):
    norm_u = np.linalg.norm(u)
    scalar_proj = np.dot(v, u) / norm_u
    proj = scalar_proj * u / norm_u
    return np.linalg.norm(proj - v)


def calc_perpendicular_dist_matrix_slow(N, ref_dirs):
    return cdist(N, ref_dirs, metric=calc_perpendicular_dist)


def calc_perpendicular_dist_matrix(N, ref_dirs):
    u = np.tile(ref_dirs, (len(N), 1))
    v = np.repeat(N, len(ref_dirs), axis=0)

    norm_u = np.linalg.norm(u, axis=1)

    scalar_proj = np.sum(v * u, axis=1) / norm_u
    proj = scalar_proj[:, None] * u / norm_u[:, None]
    val = np.linalg.norm(proj - v, axis=1)
    matrix = np.reshape(val, (len(N), len(ref_dirs)))

    return matrix
