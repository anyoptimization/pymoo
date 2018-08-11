import numpy as np


def tchebi(F, weights, ideal_point):
    v = np.abs((F - ideal_point) * weights)
    return np.max(v, axis=1)


def weighted_sum(F, weights):
    return np.sum(F * weights, axis=1)
