from pymoo.indicators.igd import IGD
from pymoo.util.normalization import normalize
from pymoo.termination.delta import DeltaToleranceTermination


class DesignSpaceTermination(DeltaToleranceTermination):

    def __init__(self, tol=0.001, **kwargs):
        super().__init__(tol, **kwargs)

    def _delta(self, prev, current):
        return IGD(current).do(prev)

    def _data(self, algorithm):

        X = algorithm.opt.get("X")

        # do normalization if bounds are given
        problem = algorithm.problem
        if X.dtype != object and problem.has_bounds():
            X = normalize(X, xl=problem.xl, xu=problem.xu)

        return X
