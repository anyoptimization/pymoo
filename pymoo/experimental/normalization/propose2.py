import numpy as np

from pymoo.model.survival import Survival, split_by_feasibility
from pymoo.operators.survival.c_reference_line_survival import c_associate_to_niches, c_get_extreme_points, \
    c_get_intercepts
from pymoo.operators.survival.reference_line_survival import niching, calc_niche_count
from pymoo.util.mathematics import Mathematics
from pymoo.util.non_dominated_sorting import NonDominatedSorting


class ProposeReferenceLineSurvival(Survival):
    def __init__(self, ref_dirs):
        super().__init__()
        self.ref_dirs = ref_dirs
        self.extreme_points = None
        self.nadir_point = None
        self.worst_point = np.full(ref_dirs.shape[1], -np.inf)
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
            self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

            # calculate the fronts of the population
            fronts, _rank = NonDominatedSorting(epsilon=Mathematics.EPS).do(F, return_rank=True,
                                                                            n_stop_if_ranked=n_survive_feasible)
            non_dominated, last_front = fronts[0], fronts[-1]

            # find the extreme points for normalization
            self.extreme_points = c_get_extreme_points(F, self.ideal_point, extreme_points=self.extreme_points)
            max_of_extremes = np.max(self.extreme_points, axis=0)

            try:

                worst_point = np.max(F, axis=0)
                nadir_point = np.max(F[non_dominated, :], axis=0)
                intercepts = c_get_intercepts(self.extreme_points, self.ideal_point, nadir_point, worst_point)
                self.nadir_point = intercepts + self.ideal_point

                #b = np.logical_or(self.nadir_point < self.ideal_point + 1e-6, self.nadir_point > self.worst_point)
                #self.nadir_point[b] = max_of_extremes[b]

                #intercepts = hyperplane_intercepts(self.extreme_points, self.ideal_point)
                #self.nadir_point = intercepts + self.ideal_point

                #b = np.logical_or(self.nadir_point < self.ideal_point + 1e-6, self.nadir_point > self.worst_point)
                #self.nadir_point[b] = max_of_extremes[b]

            except:
                self.nadir_point = max_of_extremes

            # index of the first n fronts form now on - including splitting front
            I = np.concatenate(fronts)
            F = F[I, :]

            # associate individuals to niches
            niche_of_individuals, dist_to_niche = c_associate_to_niches(F, self.ref_dirs, self.ideal_point,
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


def get_extreme_points_propose(F, ideal_point, extreme_points=None):

    # number of objectives
    n_obj = F.shape[1]

    # the extreme points updated finally
    e = np.zeros((n_obj, n_obj))

    # the weights used for a floating point issue safe implementation
    w = np.ones((n_obj, n_obj)) - np.eye(n_obj)

    # add the old extreme points to never loose them for normalization - update them basically
    _F = F
    if extreme_points is not None:
        _F = np.vstack([F, extreme_points])

    # translated individuals
    _F_prime = _F - ideal_point
    #_F_prime[_F_prime < 1e-3] = 0

    # update the extreme points for the normalization having the highest asf value each
    ASF = np.max(_F_prime * w[:, None, :], axis=2).T

    ASF_min = np.min(ASF, axis=0)[None, :] == ASF

    for m in range(n_obj):

        # indices where the asf value is minimum
        I = np.where(ASF_min[:, m])[0]

        # first break the tie by considering domination
        if len(I) > 1:

            fronts = NonDominatedSorting().do(_F[I, :])
            I = I[fronts[0]]

            # otherwise take the larger value in the m-th objective
            if len(I) > 1:
                I = I[np.argmax(_F[I, m])]

        # set the extreme point having the minimum asf value - real point not the translated one
        e[m, :] = _F[I, :]

    # find the extreme points for normalization
    tmp = c_get_extreme_points(F, ideal_point, extreme_points=extreme_points)
    if np.any(e != tmp):
        pass
        #print("sdfsdgs")

    return e


def hyperplane_intercepts(extreme_points, ideal_point):

    # prepare the gaussian elimination parameters
    M = extreme_points - ideal_point
    b = np.ones(extreme_points.shape[1])

    # find the intercepts using gaussian elimination
    plane = np.linalg.solve(M, b)

    # negative intercepts
    if not np.allclose(np.dot(M, plane), b):
        raise Exception("Gaussian Elimination failed.")

    else:

        # get the actual intercepts
        intercepts = 1 / plane

        return intercepts
