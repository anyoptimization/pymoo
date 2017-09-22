import numpy as np


class RandomFactory:
    def sample(self, n, xl, xu):
        return [np.random.random(len(xl)) * (xu - xl) + xl for _ in range(n)]
