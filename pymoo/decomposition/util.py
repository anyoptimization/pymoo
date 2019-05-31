import numpy as np


def calc_distance_to_weights(F, weights, utopian_point):
    norm = np.linalg.norm(weights, axis=1)
    F = F - utopian_point

    d1 = np.sum(F * weights, axis=1) / norm
    d2 = np.linalg.norm(F - (d1[:, None] * weights / norm[:, None]), axis=1)

    return d1, d2
