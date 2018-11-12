import numpy as np

from pymoo.cython.my_math import cython_calc_perpendicular_distance
from pymoo.rand import random


def niching(F, n_remaining, niche_count, niche_of_individuals, dist_to_niche):
    survivors = []

    # boolean array of elements that are considered for each iteration
    mask = np.full(F.shape[0], True)

    while len(survivors) < n_remaining:

        # all niches where new individuals can be assigned to
        next_niches_list = np.unique(niche_of_individuals[mask])

        # pick a niche with minimum assigned individuals - break tie if necessary
        next_niche_count = niche_count[next_niches_list]
        next_niche = np.where(next_niche_count == next_niche_count.min())[0]
        next_niche = next_niches_list[next_niche]
        next_niche = next_niche[random.randint(0, len(next_niche))]

        # indices of individuals that are considered and assign to next_niche
        next_ind = np.where(np.logical_and(niche_of_individuals == next_niche, mask))[0]

        # shuffle to break random tie (equal perp. dist) or select randomly
        next_ind = random.shuffle(next_ind)

        if niche_count[next_niche] == 0:
            next_ind = next_ind[np.argmin(dist_to_niche[next_ind])]
        else:
            # already randomized through shuffeling
            next_ind = next_ind[0]

        mask[next_ind] = False
        survivors.append(int(next_ind))

        niche_count[next_niche] += 1

    return survivors


def calc_niche_count(n_niches, niche_of_individuals):
    niche_count = np.zeros(n_niches, dtype=np.int)
    index, count = np.unique(niche_of_individuals, return_counts=True)
    niche_count[index] = count
    return niche_count


def associate_to_niches(F, niches, ideal_point, nadir_point, utopian_epsilon=0.0):
    utopian_point = ideal_point - utopian_epsilon

    # normalize by ideal point and intercepts
    N = (F - utopian_point) / (nadir_point - utopian_point)

    # dist_matrix = calc_perpendicular_dist_matrix(N, niches)
    dist_matrix = cython_calc_perpendicular_distance(N, niches)

    niche_of_individuals = np.argmin(dist_matrix, axis=1)
    dist_to_niche = dist_matrix[np.arange(F.shape[0]), niche_of_individuals]

    return niche_of_individuals, dist_to_niche
