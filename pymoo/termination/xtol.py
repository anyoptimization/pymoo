import numpy as np

from pymoo.indicators.igd import IGD
from pymoo.termination.delta import DeltaToleranceTermination
from pymoo.util.normalization import normalize


class DesignSpaceTermination(DeltaToleranceTermination):

    def __init__(self, tol=0.005, **kwargs):
        super().__init__(tol, **kwargs)

    def _delta(self, prev, current):

        if prev.dtype == float and current.dtype == float:
            return IGD(current).do(prev)
        else:
            return np.mean([np.sum(e != prev, axis=1).max() / len(e) for e in current])

    def _data(self, algorithm):

        X = algorithm.opt.get("X")

        # do normalization if bounds are given
        problem = algorithm.problem
        if X.dtype != object and problem.has_bounds():
            X = normalize(X, xl=problem.xl, xu=problem.xu)

        return X
