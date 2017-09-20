import numpy as np

from moo.util.individual import Individual


class RandomFactory:
    def __init__(self, xl, xu):
        self.xl = xl
        self.xu = xu

    def sample(self):
        ind = Individual()
        ind.x = np.random.random(len(self.xl)) * (self.xu - self.xl) + self.xl
        return ind

    def sample_more(self, n):
        return [self.sample() for _ in range(n)]
