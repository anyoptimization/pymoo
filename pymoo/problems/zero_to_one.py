import numpy as np

from pymoo.problems.meta import MetaProblem
from pymoo.util.normalization import denormalize, normalize


class ZeroToOne(MetaProblem):

    def __init__(self, problem):
        super().__init__(problem)
        assert self.xl is not None and self.xu is not None, "Both, xl and xu, must be set to redefine the problem!"

        self._xl, self._xu = problem.xl, problem.xu
        self.xl, self.xu = np.zeros(self.n_var), np.ones(self.n_var)

    def do(self, x, out, *args, **kwargs):
        out["__X__"] = x
        _x = (x - self.xl) / (self.xu - self.xl)
        super().do(_x, out, *args, **kwargs)

    def denormalize(self, x):
        return denormalize(x, self._xl, self._xu)

    def normalize(self, x):
        return normalize(x, self._xl, self._xu)


