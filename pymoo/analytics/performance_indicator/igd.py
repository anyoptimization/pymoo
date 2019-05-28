import numpy as np
from scipy.spatial.distance import cdist

from pymoo.model.indicator import Indicator


class IGD(Indicator):
    def __init__(self, pf, normalize=False):
        Indicator.__init__(self)
        self.pf = pf
        self.normalize = normalize

        if self.normalize:
            self.N = np.max(pf, axis=0) - np.min(pf, axis=0)

    def _calc(self, F):
        D = cdist(self.pf, F)
        return np.mean(np.min(D, axis=1))
