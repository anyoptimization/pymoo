import numpy as np

from pymoo.model.indicator import Indicator
from pymoo.util.misc import vectorized_cdist


def euclidean_distance(a, b, norm=None):
    return np.sqrt((((a - b) / norm) ** 2).sum(axis=1))


def modified_distance(z, a, norm=None):
    d = a - z
    d[d < 0] = 0
    d = d / norm
    return np.sqrt((d ** 2).sum(axis=1))


class DistanceIndicator(Indicator):

    def __init__(self, pf, dist_func, axis, **kwargs):
        super().__init__(pf=pf, **kwargs)
        self.dist_func = dist_func
        self.axis = axis

    def _calc(self, F):
        D = vectorized_cdist(self.pf, F, func_dist=self.dist_func, norm=self.range)
        return np.mean(np.min(D, axis=self.axis))


