import copy

import numpy as np


class Individual:

    def __init__(self,
                 X=None, F=None, G=None, H=None,
                 dF=None, dG=None, dH=None,
                 ddF=None, ddG=None, ddH=None,
                 constr_beta=None, constr_eps=0.0, constr_eq_eps=1e-6,
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

        # a set storing what has been evaluated
        self.evaluated = set()

        # the epsilon value for a solution to be considered as feasible
        self.constr_beta = constr_beta
        self.constr_eps = constr_eps
        self.constr_eq_eps = constr_eq_eps

        # additional data to be set
        self.data = kwargs
        self.attr = set(self.__dict__.keys())


    def has(self, key):
        return key in self.attr or key in self.data

    @property
    def CV(self):
        from pymoo.core.problem import calc_ieq_cv, calc_eq_cv
        ieq_cv = calc_ieq_cv(self.G, eps=self.constr_eps, beta=self.constr_beta)
        eq_cv = calc_eq_cv(self.H, eps=self.constr_eq_eps)
        return np.array([ieq_cv + eq_cv])

    @property
    def feasible(self):
        return self.CV <= 0.0

    def set(self, key, value):
        if key in self.attr:
            self.__dict__[key] = value
        else:
            self.data[key] = value
        return self

    def set_by_dict(self, **kwargs):
        for k, v in kwargs.items():
            self.set(k, v)

    def copy(self, deep=False):
        ind = copy.copy(self)
        ind.data = copy.copy(self.data) if not deep else copy.deepcopy(self.data)
        return ind

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
