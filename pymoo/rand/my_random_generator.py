import random

import numpy as np


class MyRandomGenerator:
    def __init__(self):
        self._seed = -1
        self.oldrand = np.zeros(55)
        self.jrand = 0

    def seed(self, x):
        if x > 1:
            x = x / 100.0
        self._seed = x
        self.warmup_random()


    def warmup_random(self):
        self.oldrand[54] = self._seed
        new_random = 0.000000001
        prev_random = self._seed
        for j1 in range(1,55):
            ii = (21 * j1) % 54
            self.oldrand[ii] = new_random
            new_random = prev_random - new_random
            if new_random < 0.0:
                new_random += 1.0
            prev_random = self.oldrand[ii]

        self.advance_random()
        self.advance_random()
        self.advance_random()
        self.jrand = 0


    def advance_random(self):
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

    def random(self, size=None):
        if size is None:
            return self.randomperc()          
        else:
            n = size[0]
            m = size[1]
            val = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    val[i, j] = self.randomperc()

            return val

    def randomperc(self):
        self.jrand += 1
        if self.jrand >= 55:
            self.jrand = 1
            self.advance_random()
        return self.oldrand[self.jrand]

    def permutation(self, length, n=2):

        if n != 2:
            print("Not Implemented!")

        a1 = np.array(range(length))
        a2 = np.array(range(length))

        for i in range(length):
            rand = self.randint(i, length-1)
            temp = a1[rand]
            a1[rand] = a1[i]
            a1[i] = temp

            rand = self.randint(i, length-1)
            temp = a2[rand]
            a2[rand] = a2[i]
            a2[i] = temp

        return [a1,a2]

    def randint(self, low, high):
        res = low + (self.randomperc() * (high - low + 1))
        if res > high:
            res = high
        return int(res)
