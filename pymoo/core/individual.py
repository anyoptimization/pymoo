import copy

import numpy as np


class Individual:

    def __init__(self,
                 X: np.ndarray = None,
                 F: np.ndarray = None,
                 G: np.ndarray = [],
                 H: np.ndarray = [],
                 dF: np.ndarray = None,
                 dG: np.ndarray = None,
                 dH: np.ndarray = None,
                 ddF: np.ndarray = None,
                 ddG: np.ndarray = None,
                 ddH: np.ndarray = None,
                 tcv=None,
                 **kwargs) -> None:

        # design variables
        self.X = X

        # objectives and constraint values
        self.F = F
        self.G = G
        self.H = H

        # first order derivation
        self.dF = dF
        self.dG = dG
        self.dH = dH

        # second order derivation
        self.ddF = ddF
        self.ddG = ddG
        self.ddH = ddH

        # if the constraint violation should be calculated adaptively, this can be set
        self.tcv = tcv

        # a set storing what has been evaluated
        self.evaluated = set()

        # additional data to be set
        self.data = kwargs
        self.attr = set(self.__dict__.keys())

    def has(self, key):
        return key in self.attr or key in self.data

    @property
    def CV(self):
        if self.tcv is not None:
            return self.tcv.calc(self.G, self.H)
        else:
            return self.data.get("CV")

    @property
    def feasible(self):
        return self.CV <= 0.0

    def set_by_dict(self, **kwargs):
        for k, v in kwargs.items():
            self.set(k, v)

    def copy(self, deep=False):
        ind = copy.copy(self)
        ind.data = copy.copy(self.data) if not deep else copy.deepcopy(self.data)
        return ind

    def set(self, key, value):
        if key in self.attr:
            self.__dict__[key] = value
        else:
            self.data[key] = value
        return self

    def get(self, *keys):

        def _get(key):
            if key in self.data:
                return self.data[key]
            elif key in self.attr:
                return self.__dict__[key]
            elif hasattr(self, key):
                return getattr(self, key)
            else:
                return None

        ret = []

        for key in keys:
            ret.append(_get(key))

        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)
