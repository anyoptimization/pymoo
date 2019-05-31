from pymoo.model.decomposition import Decomposition
import numpy as np


class WeightedSum(Decomposition):

    def _do(self, F, weights, **kwargs):
        return np.sum(F * weights, axis=1)
