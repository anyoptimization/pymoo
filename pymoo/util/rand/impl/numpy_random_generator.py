import numpy as np

from pymoo.rand.random_generator import RandomGenerator


class NumpyRandomGenerator(RandomGenerator):

    def _seed(self, x):
        np.random.seed(x)

    def random(self, size=None):
        return np.random.random(size)

    def randint(self, low, high=None, size=None):
        return np.random.randint(low, high=high, size=size)

    def perm(self, n):
        return np.random.permutation(n)

