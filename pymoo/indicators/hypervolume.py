import numpy as np
from pymoo.model.indicator import Indicator


def hypervolume(F):
    pass

class Hypervolume(Indicator):
    def __init__(self, reference_point):
        Indicator.__init__(self)
        self.reference_point = reference_point

    def _calc(self, F):

        if F.shape[1] == 0:
            return 0.0

        F = np.array([e for e in F if np.all(e < self.reference_point)])
        if len(F) == 0:
            return 0.0

        hv = hypervolume(F)
        return hv.compute(self.reference_point)
