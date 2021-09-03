import numpy as np

from pymoo.problems.meta import MetaProblem


class StochasticTunneling(MetaProblem):

    def __init__(self, problem, fmin, gamma):
        super().__init__(problem)
        self.fmin = fmin
        self.gamma = gamma

    def do(self, x, out, *args, **kwargs):
        super().do(x, out, *args, **kwargs)

        F = out["F"]
        out["__F__"] = F
        # out["F"] = 1 - np.exp(- self.gamma * (F - self.fmin))

        out["F"] = np.tanh(self.gamma * (F - self.fmin))




