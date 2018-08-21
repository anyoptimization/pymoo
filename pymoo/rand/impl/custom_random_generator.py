import numpy as np

from pymoo.rand.random_generator import RandomGenerator


class CustomRandomGenerator(RandomGenerator):

    def __init__(self):
        self.__seed = -1
        self.oldrand = np.zeros(55)
        self.jrand = 0

    def _seed(self, s):
        import random
        random.seed(s)
        self.__seed = random.random()

        self.__warmup_random()

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
