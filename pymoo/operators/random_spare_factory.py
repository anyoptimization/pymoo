import random

import numpy as np
from scipy.spatial.distance import pdist

from operators.random_factory import RandomFactory
from rand.default_random_generator import DefaultRandomGenerator


class RandomSpareFactory:
    def __init__(self):
        self.fails_until_dist_reduced = 200

    def sample(self, n, xl, xu, rnd=DefaultRandomGenerator()):

        failed = 0
        min_dist = np.linalg.norm(xu - xl) / 2

        X = np.zeros((n, len(xl)))
        X[0, :] = RandomFactory().sample(1, xl, xu, rnd=rnd)
        i = 1

        while i < n:

            if failed > self.fails_until_dist_reduced:
                failed = 0
                min_dist /= 1.01

            X[i, :] = RandomFactory().sample(1, xl, xu)
            dis = pdist(X[0:i + 1, :])

            if np.min(dis) >= min_dist:
                i += 1
            else:
                failed += 1

        return X
