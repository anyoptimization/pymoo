import numpy as np
from scipy.spatial.distance import cdist

from pymoo.model.measure import Measure


class IGD(Measure):
    def __init__(self, true_front):
        Measure.__init__(self)
        self.true_front = true_front

    def calc_(self, F):
        v = cdist(self.true_front, F)
        return np.mean(np.min(v, axis=1))
