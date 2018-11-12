import numpy as np

from pymoo.emo.niching import associate_to_niches, calc_niche_count, niching
from pymoo.model.survival import Survival, split_by_feasibility
from pymoo.util.mathematics import Mathematics
from pymoo.util.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize


class ProposeReferenceLineSurvival(Survival):

    def __init__(self, ref_dirs):
        super().__init__()
        self.ref_dirs = ref_dirs

        self.worst_point = np.full(ref_dirs.shape[1], -np.inf)
        self.ideal_point = np.full(ref_dirs.shape[1], np.inf)

        self.extreme_points = None
        self.nadir_point = None

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
            n_obj = F.shape[1]

            # calculate the fronts of the population
            fronts, _rank = NonDominatedSorting().do(F, return_rank=True, n_stop_if_ranked=n_survive_feasible)
            non_dominated, last_front = fronts[0], fronts[-1]

            # find or usually update the new ideal point - from feasible solutions
            self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
            self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

            # in the initial generation just set them to max - other it will not be none
            if self.extreme_points is None:
                self.extreme_points = np.full((n_obj, n_obj), np.inf)
                for m in range(n_obj):
                    self.extreme_points[m, :] = F[np.argmax(F[:, m]), :]

            self.extreme_points = update_extreme_points(F, self.extreme_points)

            self.nadir_point = np.diag(self.extreme_points)

            print(self.nadir_point)

            # index of the first n fronts form now on - including splitting front
            I = np.concatenate(fronts)
            F = F[I, :]

            # associate individuals to niches
            utopian_epsilon = 0.0
            niche_of_individuals, dist_to_niche = associate_to_niches(F, self.ref_dirs, self.ideal_point,
                                                                      self.nadir_point, utopian_epsilon=utopian_epsilon)

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



def update_extreme_points(F, extreme_points):
    par_max_rate_of_change = 100

    # number of objectives
    n_obj = F.shape[1]

    # the weights used to extract the objective for each extreme
    w = np.ones((n_obj, n_obj)) - np.eye(n_obj)
    b = [np.where(w[i, :])[0] for i in range(n_obj)]

    # decide for each objective to update or not
    for m in range(n_obj):

        # calculate the percental improvement in all objective except itself
        impr = 1 - (extreme_points[m, b[m]] - F[:, b[m]]) / extreme_points[m, b[m]]

        # calculate the minimum improvement regarding all objectives
        min_impr = np.min(impr, axis=1)

        # the change of the extreme point of we switch to a new one
        delta = 1 - np.abs((F[:, m] - extreme_points[m, m]) / extreme_points[m, m])

        # the rate in changing the extreme value and the improvement
        rate = delta / min_impr

        # if no improvement do not consider it for update
        I = np.where(min_impr > 0)[0]

        # otherwise filter if the small change in improvement does not justify the large change in extreme
        I = I[np.where(rate[I] < par_max_rate_of_change)[0]]

        # if not all solutions were filtered out
        if len(I) > 0:
            # choose the one with the best improvement
            I = I[np.argmax(min_impr[I])]
            extreme_points[m, :] = F[I, :]

    return extreme_points
