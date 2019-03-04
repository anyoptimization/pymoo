import numpy as np

from pymoo.algorithms.nsga3 import associate_to_niches, calc_niche_count, get_extreme_points_c, get_nadir_point
from pymoo.indicators.kktpm import KKTPM
from pymoo.model.survival import Survival
from pymoo.optimize import minimize
from pymoo.rand import random
from pymoo.util import plotting
from pymoo.util.non_dominated_sorting import NonDominatedSorting
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymop.factory import get_problem


class KKTPMReferenceSurvival(Survival):
    def __init__(self, ref_dirs):
        super().__init__(True)
        self.ref_dirs = ref_dirs
        self.extreme_points = None
        self.intercepts = None
        self.nadir_point = None
        self.ideal_point = np.full(ref_dirs.shape[1], np.inf)
        self.worst_point = np.full(ref_dirs.shape[1], -np.inf)

    def _do(self, pop, n_survive, D=None, **kwargs):

        # attributes to be set after the survival
        X, F = pop.get("X", "F")

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
        pop, rank, X, F = pop[I], rank[I], X[I], F[I]

        # update the front indices for the current population
        counter = 0
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                fronts[i][j] = counter
                counter += 1
        last_front = fronts[-1]

        # associate individuals to niches
        niche_of_individuals, dist_to_niche = associate_to_niches(F, self.ref_dirs, self.ideal_point, self.nadir_point)

        kktpm, _ = KKTPM(var_bounds_as_constraints=False).calc(X, problem)
        kktpm = kktpm[:, 0]

        pop.set('rank', rank, 'niche', niche_of_individuals, 'dist_to_niche', dist_to_niche, 'kktpm', kktpm)

        # if we need to select individuals to survive
        if len(pop) > n_survive:

            # if there is only one front
            if len(fronts) == 1:
                n_remaining = n_survive
                until_last_front = np.array([], dtype=np.int)
                niche_count = np.zeros(len(self.ref_dirs), dtype=np.int)

            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1])
                niche_count = calc_niche_count(len(self.ref_dirs), niche_of_individuals[until_last_front])
                n_remaining = n_survive - len(until_last_front)

            S = niching(F[last_front, :], n_remaining, niche_count, niche_of_individuals[last_front],
                        kktpm[last_front])

            survivors = np.concatenate((until_last_front, last_front[S].tolist()))
            pop = pop[survivors]

        return pop


def niching(F, n_remaining, niche_count, niche_of_individuals, kktpm):
    survivors = []

    # boolean array of elements that are considered for each iteration
    mask = np.full(F.shape[0], True)

    while len(survivors) < n_remaining:

        # all niches where new individuals can be assigned to
        next_niches_list = np.unique(niche_of_individuals[mask])

        # pick a niche with minimum assigned individuals - break tie if necessary
        next_niche_count = niche_count[next_niches_list]
        next_niche = np.where(next_nniche_count == next_niche_count.min())[0]
        next_niche = next_niches_list[next_niche]
        next_niche = next_niche[random.randint(0, len(next_niche))]

        # indices of individuals that are considered and assign to next_niche
        next_ind = np.where(np.logical_and(niche_of_individuals == next_niche, mask))[0]

        # shuffle to break random tie (equal perp. dist) or select randomly
        next_ind = random.shuffle(next_ind)

        if niche_count[next_niche] == 0:
            next_ind = next_ind[np.argmin(kktpm[next_ind])]
        else:
            # already randomized through shuffling
            next_ind = next_ind[0]

        mask[next_ind] = False
        survivors.append(int(next_ind))

        niche_count[next_niche] += 1

    return survivors


if __name__ == "__main__":
    problem = get_problem("dtlz1", n_var=7, n_obj=3)
    # create the reference directions to be used for the optimization
    ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=12).do()

    # create the pareto front for the given reference lines
    pf = problem.pareto_front(ref_dirs)

    res = minimize(problem,
                   method='nsga3',
                   method_args={
                       'pop_size': 92,
                       'ref_dirs': ref_dirs,
                       'survival': KKTPMReferenceSurvival(ref_dirs)
                   },
                   termination=('n_gen', 400),
                   pf=pf,
                   seed=31,
                   disp=True)

plotting.plot(pf, res.F, labels=["Pareto-front", "F"])
# plotting.plot(res.F, labels=["F"])
