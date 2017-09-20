import numpy as np

from moo.configuration import Configuration


def calc_crowding_distance(front):
    n = len(front)

    if n == 0:
        return []
    elif n == 1 or n == 2:
        return np.full(n, np.inf)
    else:

        cd = np.zeros(n)
        n_obj = len(front[0].f)

        # for each objective
        for j in range(n_obj):

            # sort by its objective
            sorted_idx = sorted(range(n), key=lambda x: front[x].f[j])

            # set the corner points to infinity
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf

            # get min and max for normalization
            f_min = front[sorted_idx[0]].f[j]
            f_max = front[sorted_idx[-1]].f[j]

            # calculate the normalization - avoid division by 0
            norm = f_max - f_min if f_min != f_max else Configuration.EPS

            # add up the crowding measure for all points in between
            for i in range(1, n - 1):
                cd[sorted_idx[i]] += (front[sorted_idx[i+1]].f[j] - front[sorted_idx[i-1]].f[j]) / norm

    return cd
