import numpy as np
from numpy.linalg import LinAlgError

from pymoo.cython.my_math import cython_calc_perpendicular_distance
from pymoo.model.survival import Survival, split_by_feasibility
from pymoo.operators.survival.reference_line_survival import calc_niche_count, niching
from pymoo.util.mathematics import Mathematics
from pymoo.util.non_dominated_sorting import NonDominatedSorting


class ProposeReferenceLineSurvival(Survival):
    def __init__(self, ref_dirs):
        super().__init__()
        self.ref_dirs = ref_dirs
        self.extreme_points = None
        self.intercepts = None
        self.nadir_point = None
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
            self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)

            # calculate the fronts of the population
            fronts, _rank = NonDominatedSorting(epsilon=1e-10).do(F, return_rank=True, n_stop_if_ranked=n_survive_feasible)
            non_dominated, last_front = fronts[0], fronts[-1]

            # calculate the worst point of feasible individuals
            worst_point = np.max(F, axis=0)
            # calculate the nadir point from non dominated individuals
            nadir_point = np.max(F[non_dominated, :], axis=0)

            # find the extreme points for normalization
            self.extreme_points = get_extreme_points(F, self.ideal_point, extreme_points=self.extreme_points)

            # find the intercepts for normalization and do backup if gaussian elimination fails
            self.intercepts = get_intercepts(self.extreme_points, self.ideal_point, nadir_point, worst_point)
            self.nadir_point = self.ideal_point + self.intercepts

            # index of the first n fronts form now on - including splitting front
            I = np.concatenate(fronts)
            F = F[I, :]

            # associate individuals to niches
            niche_of_individuals, dist_to_niche = associate_to_niches(F, self.ref_dirs, self.ideal_point,
                                                                      self.nadir_point)

            # if a splitting of the last front is not necessary
            if F.shape[0] == n_survive_feasible:
                _survivors = np.arange(F.shape[0])

            # otherwise we have to select using niching
            else:

                _last_front = np.arange(len(I) - len(last_front), len(I))
                _until_last_front = np.arange(0, len(I) - len(last_front))

                _survivors = []
                n_remaining = n_survive_feasible
                niche_count = np.zeros(len(self.ref_dirs), dtype=np.int)

                if len(fronts) > 1:
                    _survivors.extend(_until_last_front)
                    niche_count = calc_niche_count(len(self.ref_dirs), niche_of_individuals[_until_last_front])
                    n_remaining -= len(_until_last_front)

                S = niching(F[_last_front, :], n_remaining, niche_count, niche_of_individuals[_last_front],
                            dist_to_niche[_last_front])

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


def get_extreme_points(F, ideal_point, extreme_points=None):
    # calculate the asf which is used for the extreme point decomposition
    asf = np.eye(F.shape[1])
    asf[asf == 0] = 1e-6

    # add the old extreme points to never loose them for normalization
    _F = np.copy(F)
    if extreme_points is not None:
        _F = np.concatenate([_F, extreme_points], axis=0)

    # use __F because we substitute small values to be 0
    __F = _F - ideal_point
    __F[__F < 1e-3] = 0

    # update the extreme points for the normalization having the highest asf value each
    F_asf = np.max(__F / asf[:, None, :], axis=2)
    I = np.argmin(F_asf, axis=1)
    extreme_points = _F[I, :]

    return extreme_points


def get_intercepts(extreme_points, ideal_point, nadir_point, worst_point):
    # normalization of the points in the new space
    nadir_point -= ideal_point
    worst_point -= ideal_point

    use_nadir = False

    try:
        # find the intercepts using gaussian elimination
        plane = np.linalg.solve(extreme_points - ideal_point, np.ones(extreme_points.shape[1]))

        # if the plane
        if np.any(plane <= 1e-6):
            use_nadir = True
        else:
            intercepts = 1 / plane

    except LinAlgError:
        use_nadir = True

    if use_nadir:
        intercepts = nadir_point

    # if even that point is too small
    b = intercepts < 1e-6
    intercepts[b] = worst_point[b]

    # if also the worst point is very small we set it to a small value, to avoid division by zero
    #intercepts[intercepts < 1e-16] = 1e-16

    return intercepts


def associate_to_niches(F, niches, ideal_point, nadir_point, utopian_epsilon=1e-6):
    # normalize by ideal point and intercepts
    utopian_point = ideal_point # - utopian_epsilon
    N = (F - utopian_point) / (nadir_point - utopian_point)

    # dist_matrix = calc_perpendicular_dist_matrix(N, niches)
    dist_matrix = cython_calc_perpendicular_distance(N, niches)

    niche_of_individuals = np.argmin(dist_matrix, axis=1)
    dist_to_niche = dist_matrix[np.arange(F.shape[0]), niche_of_individuals]

    return niche_of_individuals, dist_to_niche
