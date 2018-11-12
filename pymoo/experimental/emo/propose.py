import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from pymoo.emo.niching import associate_to_niches, calc_niche_count, niching
from pymoo.model.survival import Survival, split_by_feasibility
from pymoo.util.dominator import Dominator
from pymoo.util.mathematics import Mathematics
from pymoo.util.non_dominated_sorting import NonDominatedSorting


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
            I = np.concatenate(fronts)

            # find or usually update the new ideal point - from feasible solutions
            self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
            self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

            # in the initial generation just set them to max - other it will not be none
            if self.extreme_points is None:
                self.extreme_points = np.full((n_obj, n_obj), np.inf)
                for m in range(n_obj):
                    self.extreme_points[m, :] = F[np.argmax(F[:, m]), :]
                self.nadir_point = np.copy(np.diag(self.extreme_points))

            # self.extreme_points = update_extreme_points(F[non_dominated, :], self.extreme_points, F[I, :])
            self.extreme_points = update_extreme_points_easy(self.extreme_points, F, non_dominated, I)
            self.nadir_point = np.copy(np.diag(self.extreme_points))

            # delta = np.diag(self.extreme_points) - self.nadir_point
            # nadir_point = self.nadir_point + 0.1 * delta

            # lower_bound = np.mean(F[I, :], 0)
            # self.nadir_point[nadir_point > lower_bound] = nadir_point[nadir_point > lower_bound]

            #print(self.nadir_point)

            # index of the first n fronts form now on - including splitting front

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


def outliers_iqr(ys, factor=1.5):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * factor)
    upper_bound = quartile_3 + (iqr * factor)
    return lower_bound, upper_bound



def update_extreme_points_easy(extreme_points, F, non_dom, considered):

    # number of objectives
    n_obj = F.shape[1]
    _F = F[non_dom, :]

    lower_bound = np.mean(F[considered,:], axis=0)

    # decide for each objective to update or not
    for m in range(n_obj):

        rel = np.array([Dominator.get_relation(_F[k, :], extreme_points[m, :]) for k in range(_F.shape[0])])

        e = None

        if np.any(rel == 1):
            I = np.where(rel == 1)[0]
            I = I[np.argmax(_F[I, m])]
            e = _F[I, :]

        elif np.all(rel == 0):
            I = np.argmax(_F[:, m])
            e = _F[I, :]

        if e is not None and e[m] > lower_bound[m]:
            extreme_points[m, :] = e

    return extreme_points


def update_extreme_points_easy2(extreme_points, F, non_dom, considered):
    # number of objectives
    n_obj = F.shape[1]
    _F = F[non_dom, :]

    # decide for each objective to update or not
    for m in range(n_obj):

        el = np.median(F[considered, m])
        _, eu = outliers_iqr(F[considered, m])

        if eu < el or eu > np.max(F[considered, m]):
            eu = np.max(F[considered, m])

        __F = _F[np.logical_and(_F[:, m] > el, _F[:, m] < eu), :]

        if len(__F) == 0:
            continue

        rel = np.array([Dominator.get_relation(__F[k, :], extreme_points[m, :]) for k in range(__F.shape[0])])

        e = None

        if np.any(rel == 1):
            I = np.where(rel == 1)[0]
            I = I[np.argmax(__F[I, m])]
            e = __F[I, :]

        elif np.all(rel == 0):
            I = np.argmax(__F[:, m])
            e = __F[I, :]

        if e is not None:
            extreme_points[m, :] = e

    return extreme_points


def update_extreme_points(extreme_points, F, non_dom):
    par_max_rate_in_change = 10

    # consider only non dominated points for update
    _F = F[non_dom, :]

    # number of objectives
    n_obj = F.shape[1]

    # the weights used to extract the objective for each extreme
    w = np.ones((n_obj, n_obj)) - np.eye(n_obj)

    # decide for each objective to update or not
    for obj in range(n_obj):

        e = None
        not_obj = np.where(w[obj, :] == 1)[0]

        # the change of the extreme point of we switch to a new one
        obj_delta = (_F[:, obj] - extreme_points[obj, obj]) / extreme_points[obj, obj]

        # calculate the percental improvement in all objective except itself
        not_obj_delta = (extreme_points[obj, not_obj] - _F[:, not_obj]) / extreme_points[obj, not_obj]

        # rate between objective and improvement of all other
        rate = np.abs(obj_delta / np.mean(not_obj_delta, axis=1))
        valid_rate_change = rate < par_max_rate_in_change

        _b = np.all(not_obj_delta < 0, axis=1)
        if not np.any(_b):
            _b = np.logical_and(np.any(not_obj_delta > 0, axis=1), np.any(not_obj_delta < 0, axis=1))

        # _b = np.logical_and(_b, valid_rate_change)

        if np.any(_b):
            I = np.where(_b)[0]
            I = I[np.argmax(_F[I, obj])]
            extreme_points[obj, :] = _F[I, :]

    return extreme_points


def update_extreme_points3(F, ideal_point, extreme_points, n_survive):
    par_max_factor = 2

    _extreme_points = extreme_points - ideal_point
    _F = F - ideal_point

    # number of objectives
    n_obj = F.shape[1]

    # the weights used to extract the objective for each extreme
    w = np.ones((n_obj, n_obj)) - np.eye(n_obj)
    b = [np.where(w[i, :] == 1)[0] for i in range(n_obj)]

    # decide for each objective to update or not
    for m in range(n_obj):

        rel = np.array([Dominator.get_relation(F[k, :], extreme_points[m, :]) for k in range(F.shape[0])])
        I = np.where(rel == 1)[0]

        # the relative improvement in objective m
        delta = (_F[I, m] - _extreme_points[m, m]) / _extreme_points[m, m]

        # calculate the percental improvement in all objective except itself - negative means improved
        _delta = (_F[I, b[m]] - _extreme_points[m, b[m]]) / _extreme_points[m, b[m]]

        # calculate the minimum improvement - maximum of negative values
        min_impr = np.max(_delta, axis=1)

        factor = np.abs(delta / min_impr)

        _b = min_impr < 0
        # _b = np.logical_and(_b, factor < par_max_factor)
        # _b = np.logical_and(_b, F[:, m] > lower_bound)

        I = I[np.where(_b)[0]]

        # if not all solutions were filtered out
        if len(I) > 0:
            # choose the one with the best improvement
            I = I[np.argmin(min_impr[I])]
            extreme_points[m, :] = F[I, :]

    return extreme_points


def update_extreme_points_my(extreme_points, F, non_dom):
    # number of objectives
    n_obj = F.shape[1]

    # the weights used to extract the objective for each extreme
    w = np.ones((n_obj, n_obj)) - np.eye(n_obj)
    b = [np.where(w[i, :] == 1)[0] for i in range(n_obj)]

    # lower bound of the extreme point to have no diversity loss
    lower_bound = np.mean(F, axis=0)

    # only consider non-dominated solutions from now
    F = F[non_dom, :]

    # decide for each objective to update or not
    for m in range(n_obj):

        # only consider an update if the solutions dominates the current extreme point
        rel = np.array([Dominator.get_relation(F[k, :], extreme_points[m, :]) for k in range(F.shape[0])])
        I = np.where(rel == 1)[0]

        # the change of the extreme point of we switch to a new one
        obj_delta = (F[I, m] - extreme_points[m, m]) / extreme_points[m, m]

        # calculate the percental improvement in all objective except itself
        other_delta = (extreme_points[m, b[m]] - F[I, b[m]]) / extreme_points[m, b[m]]
        # calculate the minimum improvement regarding all objectives
        other_min_impr = np.max(other_delta, axis=1)

        _b = other_min_impr < 0
        # b = np.logical_and(b, rate < par_max_rate_of_change)
        _b = np.logical_and(_b, F[I, m] > lower_bound[m])

        I = I[np.where(_b)[0]]

        # if not all solutions were filtered out
        if len(I) > 0:
            # choose the one with the best improvement
            I = I[np.argmin(min_impr[I])]
            extreme_points[m, :] = F[I, :]

    return extreme_points
