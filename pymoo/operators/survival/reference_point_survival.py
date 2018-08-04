import numpy as np

from pymoo.model.survival import Survival
from pymoo.rand import random
from pymoo.util.non_dominated_rank import NonDominatedRank


class ReferencePointSurvival(Survival):
    def __init__(self, ref_points, epsilon, weights, n_obj):
        super().__init__()
        self.n_obj = n_obj
        self.orig = ref_points
        self.ref_points = ref_points
        self.epsilon = epsilon

    def _do(self, pop, off, n_survive, return_only_index=False, **kwargs):

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

        # if the last front needs to be splitted
        n_remaining = n_survive - len(survival)

        if n_remaining > 0:

            F_min, F_max = pop.F.min(axis=0), pop.F.max(axis=0)

            dist_matrix, self.ref_points = calc_ref_dist_matrix(pop.F, self.orig, weights=data.weights,
                                                                n_obj=self.n_obj)
            point_distance_matrix = calc_dist_matrix(pop.F[last_front, :], pop.F[last_front, :], F_min=F_min,
                                                     F_max=F_max)

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


def calc_ref_dist_matrix(F, orig, weights, n_obj, F_min=None, F_max=None, return_bounds=False):
    def normalize_reference_points(F, orig, f_min, f_max):

        r_ = (orig - f_min) / (f_max - f_min)
        f_ = (F - f_min) / (f_max - f_min)

        r = np.tile(r_, (len(f_), 1))
        f = np.repeat(f_, len(r_), axis=0)
        d = np.sqrt(np.sum(weights * (f - r) ** 2, axis=1))

        d = np.reshape(d, (f_.shape[0], r_.shape[0]))

        return d, r_

    def normalize_extreme_points(F, n_obj):
        extreme = np.eye(n_obj)
        r = np.tile(extreme, (len(F), 1))
        f = np.repeat(F, len(extreme), axis=0)
        # matrix = np.sqrt(np.sum(weights * ((f - r) / (f_max - f_min)) ** 2, axis=1))
        matrix = np.sqrt(np.sum(weights * (f - r) ** 2, axis=1))
        matrix = np.reshape(matrix, (F.shape[0], extreme.shape[0]))

        return matrix, extreme

    if F_min is None:
        F_min = F.min(axis=0)
    if F_max is None:
        F_max = F.max(axis=0)
    if weights is None:
        weights = np.full(F.shape[1], 1 / F.shape[1])

    user_points, r_ = normalize_reference_points(F, orig, F_min, F_max)
    extreme_points, e_ = normalize_extreme_points(F, n_obj)

    matrix = np.concatenate((user_points, extreme_points), axis=1)
    ref_points = np.concatenate((r_, e_), axis=0)

    if return_bounds == True:
        return matrix, ref_points, F_min, F_max
    else:
        return matrix, ref_points


def calc_dist_matrix(F, ref_dirs, F_min=None, F_max=None, return_bounds=False):
    if F_min is None:
        F_min = F.min(axis=0)
    if F_max is None:
        F_max = F.max(axis=0)

    r = np.tile(ref_dirs, (len(F), 1))
    f = np.repeat(F, len(ref_dirs), axis=0)

    if not np.array_equal(F_max, F_min):
        matrix = np.sqrt(np.sum(((f - r) / (F_max - F_min)) ** 2, axis=1))
    else:
        matrix = np.sqrt(np.sum((f - r) ** 2, axis=1))

    matrix = np.reshape(matrix, (F.shape[0], ref_dirs.shape[0]))

    if return_bounds == True:
        return matrix, F_min, F_max
    else:
        return matrix
