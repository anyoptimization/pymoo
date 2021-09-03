import numpy as np

from pymoo.indicators.igd import IGD
from pymoo.util.misc import to_numpy
from pymoo.util.normalization import normalize
from pymoo.util.termination.sliding_window_termination import SlidingWindowTermination


class DesignSpaceToleranceTermination(SlidingWindowTermination):

    def __init__(self,
                 n_last=20,
                 tol=1e-6,
                 nth_gen=1,
                 n_max_gen=None,
                 n_max_evals=None,
                 **kwargs):

        super().__init__(metric_window_size=n_last,
                         data_window_size=2,
                         min_data_for_metric=2,
                         nth_gen=nth_gen,
                         n_max_gen=n_max_gen,
                         n_max_evals=n_max_evals,
                         **kwargs)
        self.tol = tol

    def _store(self, algorithm):
        problem = algorithm.problem
        X = algorithm.opt.get("X")

        if X.dtype != object:
            if problem.xl is not None and problem.xu is not None:
                X = normalize(X, xl=problem.xl, xu=problem.xu)
            return X

    def _metric(self, data):
        last, current = data[-2], data[-1]
        return IGD(current).do(last)

    def _decide(self, metrics):
        return to_numpy(metrics).mean() > self.tol
