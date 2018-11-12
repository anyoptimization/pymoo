import numpy as np

from pymoo.experimental.emo_new.nsga3_pbi import associate_to_niches
from pymoo.algorithms.nsga3 import get_nadir_point, calc_niche_count, niching
from pymoo.model.survival import Survival
from pymoo.util.non_dominated_sorting import NonDominatedSorting


class ReferenceDirectionSurvivalKeepExtreme(Survival):
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
        extreme_points = get_extreme_points(F[non_dominated], self.ideal_point)
        extreme_points = pop[non_dominated][extreme_points]

        # find the intercepts for normalization and do backup if gaussian elimination fails
        worst_of_population = np.max(F, axis=0)
        worst_of_front = np.max(F[non_dominated, :], axis=0)

        self.nadir_point = get_nadir_point(extreme_points.get("F"), self.ideal_point, self.worst_point,
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

        # keep the extremes in case they would not survive
        for extreme_point in extreme_points:
            if not np.any(pop == extreme_point):
                pop = pop.merge(np.array([extreme_point]))

        return pop


def get_extreme_points(F, ideal_point):
    # calculate the asf which is used for the extreme point decomposition
    asf = np.eye(F.shape[1])
    asf[asf == 0] = 1e6

    # use __F because we substitute small values to be 0
    __F = F - ideal_point
    __F[__F < 1e-3] = 0

    # update the extreme points for the normalization having the highest asf value each
    F_asf = np.max(__F * asf[:, None, :], axis=2)
    I = np.argmin(F_asf, axis=1)

    return I
