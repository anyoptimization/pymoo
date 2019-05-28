import numpy as np

from pymoo.model.indicator import Indicator
from pymoo.util.misc import vectorized_cdist


def modified_distance_func(z, a):
    d = a - z
    d[d < 0] = 0
    return np.sqrt((d ** 2).sum(axis=1))


class IGDPlus(Indicator):
    def __init__(self, pf):
        Indicator.__init__(self)
        self.pf = pf

    def _calc(self, F):
        D = vectorized_cdist(self.pf, F, modified_distance_func)
        return np.mean(np.min(D, axis=1))
