import numpy as np

from pymoo.model.survival import Survival
from pymoo.rand import random
from pymoo.util.misc import normalize_by_asf_interceptions
from pymoo.util.non_dominated_rank import NonDominatedRank
from pymoo.util.reference_directions import get_ref_dirs_from_points


class ReferenceLineSurvival(Survival):

    def __init__(self, ref_dirs, n_obj):
        super().__init__()
        self.ref_dirs = ref_dirs
        self.n_obj = n_obj
        self.extreme = None
        self.asf = None

    def _do(self, pop, n_survive, return_only_index=False, **kwargs):

        data = kwargs['data']

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

        N, self.asf, self.extreme, F_min, F_max = normalize_by_asf_interceptions(np.vstack((pop.F, data.ref_points)),
                                                                                 len(fronts[0]),
                                                                                 prev_asf=self.asf,
                                                                                 prev_S=self.extreme,
                                                                                 return_bounds=True)

        z_ = (data.ref_points - F_min)/(F_max - F_min)  # Normalized reference points
        data.F_min, data.F_max = F_min, F_max
        self.ref_dirs = get_ref_dirs_from_points(z_, self.n_obj, data.ref_pop_size, alpha=data.mu, method=data.method, p=data.p)
        data.ref_dirs = self.ref_dirs

        # if the last front needs to be split
        n_remaining = n_survive - len(survival)
        if n_remaining > 0:

            dist_matrix = calc_perpendicular_dist_matrix(N, self.ref_dirs)
            niche_of_individuals = np.argmin(dist_matrix, axis=1)
            dist_to_niche = dist_matrix[np.arange(len(dist_matrix)), niche_of_individuals]

            # for each reference direction the niche count
            niche_count = np.zeros(len(self.ref_dirs))
            for i in niche_of_individuals[survival]:
                niche_count[i] += 1

            # relative index to dist and the niches just of the last front
            dist_to_niche = dist_to_niche[last_front]
            niche_of_individuals = niche_of_individuals[last_front]

            # boolean array of elements that are considered for each iteration
            remaining_last_front = np.full(len(last_front), True)

            while n_remaining > 0:

                # all niches where new individuals can be assigned to
                next_niches_list = np.unique(niche_of_individuals[remaining_last_front])

                # pick a niche with minimum assigned individuals - break tie if necessary
                next_niche_count = niche_count[next_niches_list]
                next_niche = np.where(next_niche_count == next_niche_count.min())[0]
                next_niche = next_niche[random.randint(0, len(next_niche))]
                next_niche = next_niches_list[next_niche]

                # indices of individuals that are considered and assign to next_niche
                next_ind = np.where(np.logical_and(niche_of_individuals == next_niche, remaining_last_front))[0]

                if len(next_ind) == 1:
                    next_ind = next_ind[0]
                elif niche_count[next_niche] == 0:
                    next_ind = next_ind[np.argmin(dist_to_niche[next_ind])]
                else:
                    next_ind = next_ind[random.randint(0, len(next_ind))]

                remaining_last_front[next_ind] = False
                survival.append(last_front[next_ind])

                niche_count[next_niche] += 1
                n_remaining -= 1

        if return_only_index:
            return survival

        # now truncate the population
        pop.filter(survival)
        # np.savetxt("ref_dirs.txt", data.ref_dirs)
        return pop

def calc_perpendicular_dist_matrix(N, ref_dirs):
    u = np.tile(ref_dirs, (len(N), 1))
    v = np.repeat(N, len(ref_dirs), axis=0)

    norm_u = np.linalg.norm(u, axis=1)

    scalar_proj = np.sum(v * u, axis=1) / norm_u
    proj = scalar_proj[:, None] * u / norm_u[:, None]
    val = np.linalg.norm(proj - v, axis=1)
    matrix = np.reshape(val, (len(N), len(ref_dirs)))

    return matrix
