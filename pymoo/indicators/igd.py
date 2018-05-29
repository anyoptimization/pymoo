import numpy as np
from scipy.spatial.distance import cdist

from pymoo.model.indicator import Indicator


class IGD(Indicator):
    def __init__(self, true_front):
        Indicator.__init__(self)
        self.true_front = true_front

    def _calc(self, F):
        v = cdist(self.true_front, F)
        return np.mean(np.min(v, axis=1))
