import numpy as np


class Population:
    def __init__(self, X=None, F=None, G=None):
        self.X = X  # design variables
        self.F = F  # objective values
        self.G = G  # constraint violations as vectors

    def merge(self, other):
        self.X = np.concatenate([self.X, other.X])
        self.F = np.concatenate([self.F, other.F])
        self.G = np.concatenate([self.G, other.G])
        return self

    def size(self):
        if self.X is None:
            return 0
        else:
            return len(self.X)

    def filter(self, v):
        if self.X is None:
            return 0
        else:
            self.X = self.X[v]
            self.F = self.F[v]
            self.G = self.G[v]
