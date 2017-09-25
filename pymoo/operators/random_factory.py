import random

import numpy as np


class RandomFactory:
    def sample(self, n, xl, xu):
        val = np.zeros((n, len(xl)))

        for i in range(n):
            val[i, :] = random.uniform(xl, xu)

        return val
        # return np.array([np.random.uniform(xl[i], xu[i], n) for i in range(len(m))]).T
