import numpy as np
from scipy.spatial.distance import cdist

from pymoo.model.indicator import Indicator


class GD(Indicator):
    def __init__(self, pf):
        Indicator.__init__(self)
        self.pf = pf

    def _calc(self, F):
        D = cdist(self.pf, F)
        return np.mean(np.min(D, axis=0))
