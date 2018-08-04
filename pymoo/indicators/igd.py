import numpy as np
from scipy.spatial.distance import cdist

from pymoo.model.indicator import Indicator


class IGD(Indicator):
    def __init__(self, pf):
        Indicator.__init__(self)
        self.pf = pf

    def _calc(self, F):
        v = cdist(self.pf, F)
        return np.mean(np.min(v, axis=1))
