import numpy as np

class Population:
    def __init__(self):
        self.X = None  # design variables
        self.F = None  # objective values
        self.G = None  # constraint violations as vectors


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

            X = np.zeros((len(v), self.X.shape[1]))
            F = np.zeros((len(v), self.F.shape[1]))
            G = np.zeros((len(v), self.G.shape[1]))

            for i, o in enumerate(v):
                X[i,:] = self.X[o,:]
                F[i, :] = self.F[o, :]
                G[i, :] = self.G[o, :]

            self.X = X
            self.F = F
            self.G = G

