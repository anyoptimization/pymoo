import numpy as np

from pymoo.model.survival import Survival, split_by_feasibility
from pymoo.util.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort


class RankAndCrowdingSurvival(Survival):
    def _do(self, pop, n_survive, out=None, **kwargs):

        # split by feasibility
        feasible, infeasible = split_by_feasibility(pop)

        # final result that contains indices, rank and crowding of surviving individuals
        survivors = []
        crowding = []
        rank = []

        if len(feasible) > 0:

            # calculate rank only of feasible solutions
            F = pop.F[feasible, :]
            fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

            # go through all fronts except the last one
            for k, front in enumerate(fronts):

                # calculate the crowding distance of the front
                #crowding_of_front = calc_crowding_distance(F[front, :])
                crowding_of_front = calc_crowding_distance_vectorized(F[front, :])

                # current front sorted by crowding distance if splitting
                if len(survivors) + len(front) > n_survive:
                    I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                    I = I[:(n_survive - len(survivors))]

                # otherwise take the whole front
                else:
                    I = np.arange(len(front))

                # calculate crowding distance for the current front
                crowding.append(crowding_of_front[I])
                rank.append(np.array([k] * len(I)))
                survivors.extend(front[I])

            # create numpy arrays out of the lists
            rank = np.concatenate(rank)
            crowding = np.concatenate(crowding)

            # get absolute index from filtering before
            survivors = feasible[survivors]

        # if infeasible solutions need to be added - individuals sorted by constraint violation are added
        n_infeasible = (n_survive - len(survivors))
        if n_infeasible > 0:
            survivors = np.concatenate([survivors, infeasible[:n_infeasible]])
            rank = np.concatenate([rank, 1e16 * np.ones(n_infeasible)])
            crowding = np.concatenate([crowding, -1.0 * np.ones(n_infeasible)])

        # now truncate the population
        pop.filter(survivors)

        if out is not None:
            out['rank'] = rank
            out['crowding'] = crowding


def calc_crowding_distance_vectorized(F):
    infinity = 1e+14

    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        return np.full(n_points, infinity)
    else:

        # the final crowding distance result
        crowding = np.zeros(n_points)

        # sort each column and get index
        I = np.argsort(F, axis=0, kind='mergesort')

        # now really sort the whole array
        F = F[I, np.arange(n_obj)]

        # get the distance to the last element in sorted list and replace zeros with actual values
        dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) \
               - np.concatenate([np.full((1, n_obj), -np.inf), F])

        index_dist_is_zero = np.where(dist == 0)

        dist_to_last = np.copy(dist)
        for i, j in zip(*index_dist_is_zero):
            dist_to_last[i, j] = dist_to_last[i - 1, j]

        dist_to_next = np.copy(dist)
        for i, j in reversed(list(zip(*index_dist_is_zero))):
            dist_to_next[i, j] = dist_to_next[i + 1, j]

        # normalize all the distances
        norm = np.max(F, axis=0) - np.min(F, axis=0)
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

    # replace infinity with a large number
    crowding[np.isinf(crowding)] = infinity

    return crowding


def calc_crowding_distance(F):
    infinity = 1e+14

    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        return np.full(n_points, infinity)
    else:

        # the final crowding distance result
        crowding = np.zeros(n_points)

        # for each objective
        for m in range(n_obj):

            # sort by objective randomize if they are equal
            I = np.argsort(F[:, m], kind='mergesort')
            # I = randomized_argsort(F[:, m], order='ascending')

            # norm which will be used for distance normalization
            norm = np.max(F[:, m]) - np.min(F[:, m])

            # set crowding to infinity of extreme point
            crowding[I[0]] = np.inf
            crowding[I[-1]] = np.inf

            # if norm is zero -> next objective
            if norm != 0.0:

                # add up the crowding measure for all points in between
                for i in range(1, n_points - 1):

                    # the current values to have a look at
                    _current, _last, _next = i, i - 1, i + 1

                    # if the current entry is already infinity the values will not change
                    if not np.isinf(crowding[I[_current]]):
                        # crowding[I[_current]] += (F[I[_next], m] - F[I[_last], m]) / norm

                        # search for last and next value that are not equal
                        while _last >= 0 and F[I[_last], m] == F[I[_current], m]:
                            _last -= 1

                        while _next < n_points and F[I[_next], m] == F[I[_current], m]:
                            _next += 1

                        # if the point is in fact also an extreme point
                        if _last < 0 or _next == n_points:
                            crowding[I[_current]] = np.inf

                        # otherwise, which will be usually the case
                        else:
                            crowding[I[_current]] += (F[I[_next], m] - F[I[_last], m]) / norm

    # divide by the number of objectives
    crowding = crowding / n_obj

    # replace infinity with a large number
    crowding[np.isinf(crowding)] = infinity

    return crowding


if __name__ == '__main__':
    F = np.random.rand(100, 2)
    F[0] = np.max(F, axis=0)
    F[5] = F[0]
    F[50] = F[51]

    crowding = calc_crowding_distance_vectorized(F)
    print(crowding[50], crowding[51])
    print(crowding[0], crowding[5])
