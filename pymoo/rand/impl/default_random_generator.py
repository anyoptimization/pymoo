import random
import numpy as np


class DefaultRandomGenerator:

    def random(self, size=None):
        if size is None:
            return random.random()
        else:
            n = size[0]
            m = size[1]
            val = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    val[i, j] = random.random()

            return val


    def permutation(self, length, n=1):
        perms = []
        for _ in range(n):
            perm = list(range(length))
            random.shuffle(perm)
            perms.append(perm)
        return perms

    def seed(self, x):
        random.seed(x)

    def randint(self, low, high):
        if low >= high:
            return low
        else:
            res = low + (self.random() * (high - low))
            if res > high:
                res = high
        return int(res)

