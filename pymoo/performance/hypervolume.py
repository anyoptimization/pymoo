import numpy as np
from pymoo.model.performance import Performance


def hypervolume(F):
    pass

class Hypervolume(Performance):
    def __init__(self, reference_point):
        Performance.__init__(self)
        self.reference_point = reference_point

    def _calc(self, F):

        if F.shape[1] == 0:
            return 0.0

        F = np.array([e for e in F if np.all(e < self.reference_point)])
        if len(F) == 0:
            return 0.0

        hv = hypervolume(F)
        return hv.compute(self.reference_point)
