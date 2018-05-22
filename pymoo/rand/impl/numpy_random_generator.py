import numpy as np

from pymoo.rand.random_generator import RandomGenerator


class NumpyRandomGenerator(RandomGenerator):

    def seed(self, x):
        np.random.seed(x)

    def rand(self, size=None):
        if size is None:
            return np.random.random()
        elif isinstance(size, int):
            return np.random.random(size)
        else:
            return np.random.random((size[0], size[1]))

    def randint(self, low, high, size=None):
        return np.random.randint(low, high, size)

    def perm(self, n):
        return np.random.permutation(n)

