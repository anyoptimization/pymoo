import numpy as np

from pymoo.core.meta import Meta
from pymoo.core.problem import Problem
from pymoo.util.normalization import denormalize, normalize


class ZeroToOne(Meta, Problem):

    def __init__(self, problem):
        super().__init__(problem)
        assert self.xl is not None and self.xu is not None, "Both, xl and xu, must be set to redefine the problem!"
        self.xl, self.xu = np.zeros(self.n_var), np.ones(self.n_var)

    def do(self, X, return_values_of, *args, **kwargs):
        Xp = self.normalize(X)

        out = super().do(Xp, return_values_of, *args, **kwargs)
        out["__X__"] = X

        return out

    def denormalize(self, x):
        return denormalize(x, self.__object__.xl, self.__object__.xu)

    def normalize(self, x):
        return normalize(x, self.__object__.xl, self.__object__.xu)
