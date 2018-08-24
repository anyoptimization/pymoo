import copy

import numpy as np


class Population:

    def __init__(self, **kwargs):
        self.D = dict(kwargs)

    def __getattr__(self, name):

        # if internal function method just return the one whats usually done
        if str.startswith(name, '__'):
            return super().__getattr__(name)

        # if we really ask for the dictionary
        if name == "D":
            return self.__dict__.get("D", None)
        # else we ask for an entry in the dictionary
        else:
            if "D" in self.__dict__:
                return self.__dict__["D"].get(name, None)
            else:
                print(name)
                return {}

    def __setattr__(self, name, value):
        if name == 'D':
            self.__dict__['D'] = value
        else:
            self.__dict__['D'][name] = value

    def __deepcopy__(self, a):
        return Population(**copy.deepcopy(self.D))

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

        # all values in other that were not present in self
        for key, value in other.D.items():
            if key not in self.D:
                D[key] = other.D[key]

        self.D = D

    def size(self):
        if self.X is not None:
            return self.X.shape[0]
        elif self.F is not None:
            return self.F.shape[0]
        else:
            return None

    def copy(self):
        pop = Population()
        pop.D = copy.deepcopy(self.D)
        return pop

    def filter(self, v):

        if isinstance(v, np.ndarray):
            v = v.astype(np.int)

        D = {}
        for key, value in self.D.items():
            if value is not None and isinstance(value, np.ndarray):
                D[key] = value[v, :]
            else:
                D[key] = value
        self.D = D
