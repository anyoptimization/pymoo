import numpy as np

from pymoo.indicators.igd import IGD
from pymoo.termination.delta import DeltaToleranceTermination
from pymoo.util.normalization import normalize



class DesignSpaceTermination(DeltaToleranceTermination):

    def __init__(self, tol=0.005, **kwargs):
        """
        Check the distance in the design-space and terminate based on tolerance.
        (only works if variables can be converted to float)
        """
        super().__init__(tol, **kwargs)

    def _delta(self, prev, current):
        try:
            return IGD(current.astype(float)).do(prev.astype(float))
        except:
            return np.inf

    def _data(self, algorithm):

        X = algorithm.opt.get("X")

        # do normalization if bounds are given
        problem = algorithm.problem
        if X.dtype != object and problem.has_bounds():
            X = normalize(X, xl=problem.xl, xu=problem.xu)

        return X
