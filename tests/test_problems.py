import numpy as np

from pymop.problem import Problem


class AlwaysInfeasibleProblem(Problem):

    def __init__(self):
        Problem.__init__(self)
        self.n_var = 1
        self.n_constr = 1
        self.n_obj = 1
        self.func = self._evaluate
        self.xl = np.array([0])
        self.xu = np.array([1])

    def _evaluate(self, x, f, g):
        g[:, 0] = 1
