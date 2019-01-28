from abc import abstractmethod

import numpy as np


class RandomGenerator:
    """
    Implementation of a random generator used for all algorithm. This is just the base class which needs to
    be inherited from.
    """

    def seed(self, n):
        if n < 0:
            raise Exception("Random seed must be larger than 0!")
        self._seed(n)

    def perm(self, size):
        return np.argsort(self.random(size), kind='quicksort')

    def randint(self, low, high=None, size=None):
        val = self.random(size)
        return (low + (val * (high - low))).astype(np.int)

    def choice(self, a):
        return a[self.randint(0, len(a))]

    def shuffle(self, a):
        return a[self.perm(a.shape[0])]

    def random(self, size=None):
        if size is None:
            return self._rand((1, 1))[0, 0]

        elif isinstance(size, int):
            return self._rand((1, size))[0, :]

        elif isinstance(size, tuple):
            return self._rand(size)
        else:
            raise Exception("rand method not defined for: %s" % size)

    def _rand(self, size=None):
        val = np.full(size, np.inf)
        for index, _ in np.ndenumerate(val):
            val[index] = self._rand_float()
        return val

    @abstractmethod
    def _rand_float(self):
        pass

    @abstractmethod
    def _seed(self, s):
        pass
