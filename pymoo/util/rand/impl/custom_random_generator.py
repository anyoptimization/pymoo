import numpy as np

from pymoo.rand.random_generator import RandomGenerator
from pymoo.util.misc import swap


class CustomRandomGenerator(RandomGenerator):

    def __init__(self, seed=None):
        self.__seed = seed
        self._seed(self.__seed)

    def _seed(self, s):

        if s is None:
            import random
            self.__seed = random.random()
        elif s >= 1.0:
            import random
            random.seed(int(s))
            self.__seed = random.random()
        else:
            self.__seed = s

        self.oldrand = np.zeros(55)
        self.jrand = 0
        self.__warmup_random()

    def _rand(self, size=None):
        val = np.full(size, np.inf)
        for index, _ in np.ndenumerate(val):
            val[index] = self._rand_float()
        return val

    def perm(self, size):
        a = np.arange(size)
        for i in range(size):
            rnd = self.randint(i, size)
            swap(a, rnd, i)
        return a

    def _rand_float(self):
        self.jrand += 1
        if self.jrand >= 55:
            self.jrand = 1
            self.__advance_random()
        return self.oldrand[self.jrand]

    def __warmup_random(self):
        self.oldrand[54] = self.__seed
        new_random = 0.000000001
        prev_random = self.__seed
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
