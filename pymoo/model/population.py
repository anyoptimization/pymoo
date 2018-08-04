import numpy as np


class Population:

    def __init__(self, **kwargs):
        D = dict(kwargs)
        self.D = D

    def __getattr__(self, name):
        if name == "D":
            return self.D
        else:
            return self.D[name]

    def __setattr__(self, name, value):
        if name == "D":
            self.__dict__[name] = value
        else:
            self.D[name] = value

    def merge(self, other):
        D = {}
        for key, value in self.D.items():

            # key must be in both populations
            if key not in other.D:
                continue

            if isinstance(value, np.ndarray):
                try:
                    D[key] = np.concatenate([self.D[key], other.D[key]])
                except:
                    D[key] = self.D[key]
        self.D = D

    def size(self):
        return self.X.shape[0]

    def filter(self, v):
        D = {}
        for key, value in self.D.items():
            if value is not None and isinstance(value, np.ndarray):
                D[key] = value[v, :]
            else:
                D[key] = value
        self.D = D
