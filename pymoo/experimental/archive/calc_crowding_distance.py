
import numpy as np

from pymoo.util.mathematics import Mathematics


def calc_crowding_distance(F):

    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        return np.full(n_points, Mathematics.INF)
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
    crowding[np.isinf(crowding)] = Mathematics.INF

    return crowding

