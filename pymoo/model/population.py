import numpy as np


class Population:
    def __init__(self, X=None, F=None, G=None):
        self.X = X  # design variables
        self.F = F  # objective values
        self.G = G  # constraint violations as vectors

    def merge(self, other):
        if self.X is not None:
            self.X = np.concatenate([self.X, other.X])
        else:
            self.X = other.X

        if self.F is not None:
            self.F = np.concatenate([self.F, other.F])
        else:
            self.F = other.F

        if self.G is not None and other.G is not None:
            self.G = np.concatenate([self.G, other.G])
        else:
            self.G = other.G

        return self

    def size(self):
        if self.F is None:
            return 0
        else:
            return len(self.F)

    def filter(self, v):
        if self.X is None:
            return 0
        else:
            self.X = self.X[v]
            self.F = self.F[v]
            if self.G is not None:
                self.G = self.G[v]
        return self
