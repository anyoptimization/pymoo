import numpy as np


class NumpyRandomGenerator:

    def random(self, size=None):
        if size is None:
            return np.random.random()
        else:
            return np.random.random((size[0], size[1]))

    def permutation(self, n):
        return np.random.permutation(n)

    def seed(self, x):
        np.random.seed(x)
