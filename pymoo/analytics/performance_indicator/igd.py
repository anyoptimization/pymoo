import numpy as np
from scipy.spatial.distance import cdist

from pymoo.model.indicator import Indicator
from pymoo.util.misc import vectorized_cdist


class IGD(Indicator):
    def __init__(self, pareto_front, normalize=False):
        Indicator.__init__(self)
        self.pareto_front = pareto_front
        self.normalize = normalize

        if self.normalize:
            self.N = np.max(pareto_front, axis=0) - np.min(pareto_front, axis=0)

    def _calc(self, F):

        if self.normalize:
            def dist(A, B):
                return np.sqrt(np.sum(np.square((A - B) / self.N), axis=1))

            D = vectorized_cdist(self.pareto_front, F, dist)

        else:
            D = cdist(self.pareto_front, F)

        return np.mean(np.min(D, axis=1))
