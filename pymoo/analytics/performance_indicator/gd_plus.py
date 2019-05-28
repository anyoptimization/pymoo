import numpy as np

from pymoo.analytics.performance_indicator.igd_plus import modified_distance_func
from pymoo.model.indicator import Indicator
from pymoo.util.misc import vectorized_cdist


class GDPlus(Indicator):
    def __init__(self, pf):
        Indicator.__init__(self)
        self.pf = pf

    def _calc(self, F):
        D = vectorized_cdist(self.pf, F, modified_distance_func)
        return np.mean(np.min(D, axis=0))
