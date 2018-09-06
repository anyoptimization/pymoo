import numpy as np


class Individual:

    def __init__(self) -> None:
        self.X = None
        self.D = {}

    def get_genome(self):
        return self.X

    def set_genome(self, X):
        self.X = X


def get_genome(individuals):
    n_individuals = len(individuals)
    n_dim = individuals[0, 0].get_genome().shape[0]
    X = np.full((n_individuals, n_dim), np.inf, dtype=np.double)
    for i in range(n_individuals):
        X[i, :] = individuals[i, 0].get_genome()
    return X


def create_from_genome(clazz, X):
    if clazz is None:
        return np.full((X.shape[0], 1), np.inf)
    else:
        l = []
        for i in range(X.shape[0]):
            obj = clazz()
            obj.set_genome(X[i, :])
            l.append(obj)
        return np.array(l, dtype=np.object)[:, None]


def create_as_objects(clazz, n_objects):
    if clazz is None:
        return np.full((n_objects, 1), np.inf)
    else:
        l = []
        for i in range(n_objects):
            l.append(clazz())
        return np.array(l, dtype=np.object)[:, None]
