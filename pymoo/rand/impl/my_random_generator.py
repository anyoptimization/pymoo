import numpy as np

from pymoo.rand.random_generator import RandomGenerator


class MyRandomGenerator(RandomGenerator):

    def __init__(self):
        self._seed = -1
        self.oldrand = np.zeros(55)
        self.jrand = 0

    def seed(self, s):
        import random
        random.seed(s)
        self._seed = random.random()

        self.__warmup_random()

    def rand(self, size=None):

        if isinstance(size, tuple) and len(size) == 1:
            size = size[0]

        if size is None:
            return self.__randomperc()
        elif isinstance(size, int):
            val = np.zeros(size)
            for j in range(size):
                val[j] = self.__randomperc()
            return val
        elif isinstance(size, tuple) and len(size) == 2:

            n = size[0]
            m = size[1]
            val = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    val[i, j] = self.__randomperc()

            return val
        else:
            raise Exception("Random Generator size % is not provided." % size)

    def perm(self, n):
        a = np.array(range(n))
        for i in range(n):
            rand = self.randint(i, n - 1)
            temp = a[rand]
            a[rand] = a[i]
            a[i] = temp
        return a

    def randint(self, low, high, size=None):
        val = self.rand(size=size)
        return (low + (val * (high - low))).astype(np.int)

    def __warmup_random(self):
        self.oldrand[54] = self._seed
        new_random = 0.000000001
        prev_random = self._seed
        for j1 in range(1, 55):
            ii = (21 * j1) % 54
            self.oldrand[ii] = new_random
            new_random = prev_random - new_random
            if new_random < 0.0:
                new_random += 1.0
            prev_random = self.oldrand[ii]

        self.__advance_random()
        self.__advance_random()
        self.__advance_random()
        self.jrand = 0

    def __randomperc(self):
        self.jrand += 1
        if self.jrand >= 55:
            self.jrand = 1
            self.__advance_random()
        return self.oldrand[self.jrand]

    def __advance_random(self):
        for j1 in range(24):
            new_random = self.oldrand[j1] - self.oldrand[j1 + 31]
            if new_random < 0.0:
                new_random = new_random + 1.0
            self.oldrand[j1] = new_random

        for j1 in range(24, 55):
            new_random = self.oldrand[j1] - self.oldrand[j1 - 24]
            if new_random < 0.0:
                new_random = new_random + 1.0
            self.oldrand[j1] = new_random

