import numpy as np

from pymoo.algorithms.nsga3 import NSGA3
from pymoo.model.survival import Survival, split_by_feasibility
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.survival.reference_line_survival import get_extreme_points, get_intercepts, associate_to_niches, \
    calc_niche_count, niching
from pymoo.util.mathematics import Mathematics
from pymoo.util.non_dominated_sorting import NonDominatedSorting


class AdaptiveNSGA3(NSGA3):

    def __init__(self, **kwargs):
        set_if_none(kwargs, 'survival', AdaptiveReferenceLineSurvival(kwargs['ref_dirs']))
        super().__init__(**kwargs)


class AdaptiveReferenceLineSurvival(Survival):
    def __init__(self, ref_dirs):
        super().__init__()
        self.ref_dirs = ref_dirs
        self.extreme_points = None
        self.intercepts = None
        self.ideal_point = np.full(ref_dirs.shape[1], np.inf)

    def _do(self, pop, n_survive, D=None, **kwargs):

        # convert to integer for later usage
        n_survive = int(n_survive)

        # first split by feasibility for normalization
        feasible, infeasible = split_by_feasibility(pop)

        # number of survivors from the feasible population
        # in case of having not enough feasible solution all feasible will survive
        if len(feasible) < n_survive:
            n_survive_feasible = len(feasible)
        else:
            n_survive_feasible = n_survive

        # attributes to be set after the survival
        survivors, rank, niche_of_individuals, dist_to_niche = [], [], [], []

        # if there are feasible solutions to survive
        if len(feasible) > 0:

            # consider only feasible solutions form now on
            F = pop.F[feasible, :]

            # find or usually update the new ideal point - from feasible solutions
            self.ideal_point = np.min(np.concatenate([self.ideal_point[None, :], F], axis=0), axis=0)

            # calculate the fronts of the population
            fronts, _rank = NonDominatedSorting(epsilon=Mathematics.EPS).do(F, return_rank=True,
                                                                            n_stop_if_ranked=n_survive_feasible)
            non_dominated, last_front = fronts[0], fronts[-1]

            # calculate the worst point of feasible individuals
            worst_point = np.max(F, axis=0)
            # calculate the nadir point from non dominated individuals
            nadir_point = np.max(F[non_dominated, :], axis=0)

            # find the extreme points for normalization
            self.extreme_points = get_extreme_points(F, self.ideal_point, extreme_points=self.extreme_points)

            # find the intercepts for normalization and do backup if gaussian elimination fails
            self.intercepts = get_intercepts(self.extreme_points, self.ideal_point, nadir_point, worst_point)
            # self.intercepts = np.max(F[I_ideal, :], axis=0) - self.ideal_point

            # now add first all points that contribute to the ideal point and extremes
            I_ideal = np.argmin(F, axis=0)
            I_extreme = get_extreme_points_indices(F, self.ideal_point)
            I_survive = np.unique(np.concatenate([I_ideal, I_extreme]))

            survivors.append(feasible[I_survive])

            counter = len(survivors)
            _fronts = []
            for front in fronts:
                front = np.array([i for i in front if i not in I_survive])

                if len(front) == 0:
                    continue

                _fronts.append(front)
                counter += len(front)

                if counter >= n_survive_feasible:
                    break

            fronts = _fronts
            last_front = fronts[-1]

            _niche_of_individuals, _ = associate_to_niches(F[I_survive, :], self.ref_dirs, self.ideal_point,
                                                           self.intercepts)
            _niche_count = calc_niche_count(len(self.ref_dirs), _niche_of_individuals)

            n_survive_feasible -= len(I_survive)
            print(I_survive)

            # index of the first n fronts form now on - including splitting front
            I = np.concatenate(fronts)
            F = F[I, :]

            # associate individuals to niches
            niche_of_individuals, dist_to_niche = associate_to_niches(F, self.ref_dirs, self.ideal_point,
                                                                      self.intercepts)

            # if a splitting of the last front is not necessary
            if F.shape[0] == n_survive_feasible:
                _survivors = np.arange(F.shape[0])

            # otherwise we have to select using niching
            else:

                _last_front = np.arange(len(I) - len(last_front), len(I))
                _until_last_front = np.arange(0, len(I) - len(last_front))

                _survivors = []
                n_remaining = n_survive_feasible
                niche_count = _niche_count

                if len(fronts) > 1:
                    _survivors.extend(_until_last_front)
                    niche_count = calc_niche_count(len(self.ref_dirs), niche_of_individuals[_survivors]) + _niche_count
                    n_remaining -= len(_survivors)

                S = niching(F[_last_front, :], n_remaining, niche_count, niche_of_individuals[_last_front],
                            dist_to_niche[_last_front])

                # S = niching_vectorized(F[_last_front, :], n_remaining, niche_count, niche_of_individuals[_last_front],
                #                       dist_to_niche[_last_front])

                _survivors.extend(_last_front[S].tolist())

            # reindex the survivors to the absolute index
            survivors = feasible[I[_survivors]]

            # save the attributes for surviving individuals
            rank = _rank[I[_survivors]]
            niche_of_individuals = niche_of_individuals[_survivors]
            dist_to_niche = dist_to_niche[_survivors]

        # if we need to fill up with infeasible solutions - we do so. Also, the data structured need to be reindexed
        n_infeasible = n_survive - len(survivors)
        if n_infeasible > 0:
            survivors = np.concatenate([survivors, infeasible[:n_infeasible]])
            rank = np.concatenate([rank, Mathematics.INF * np.ones(n_infeasible)])
            niche_of_individuals = np.concatenate([niche_of_individuals, -1 * np.ones(n_infeasible)])
            dist_to_niche = np.concatenate([dist_to_niche, Mathematics.INF * np.ones(n_infeasible)])

        # set attributes globally for other modules
        if D is not None:
            D['rank'] = rank
            D['niche'] = niche_of_individuals
            D['dist_to_niche'] = dist_to_niche

        # now truncate the population
        pop.filter(survivors)


def get_extreme_points_indices(F, ideal_point):
    # calculate the asf which is used for the extreme point decomposition
    asf = np.eye(F.shape[1])
    asf[asf == 0] = 1e-6

    # update the extreme points for the normalization having the highest asf value each
    F_asf = np.max((F - ideal_point) / asf[:, None, :], axis=2)
    return np.argmin(F_asf, axis=1)
