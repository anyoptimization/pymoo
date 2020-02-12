import numpy as np

from pymoo.performance_indicator.igd import IGD
from pymoo.util.normalization import normalize
from pymoo.util.termination.tolerance import ToleranceBasedTermination


class DesignSpaceToleranceTermination(ToleranceBasedTermination):

    def __init__(self, **kwargs) -> None:
        super().__init__(n_hist_at_least=2, n_hist=2, **kwargs)

    def _store(self, algorithm):
        X = algorithm.opt.get("X")
        if X.dtype != np.object:
            return normalize(algorithm.opt.get("X"), x_min=algorithm.problem.xl, x_max=algorithm.problem.xu)

    def _calc_metric(self):
        last, current = self.history[0], self.history[1]
        if last is not None and current is not None:
            return IGD(current).calc(last)

    def _decide(self):
        if any([e is None for e in self.metrics]):
            return True
        else:
            return np.array(self.metrics[-self.n_last:]).std() > self.tol
