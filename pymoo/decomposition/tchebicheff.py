import numpy as np

from pymoo.core.decomposition import Decomposition


class Tchebicheff(Decomposition):

    def _do(self, F, weights, **kwargs):
        v = np.abs(F - self.utopian_point) * weights
        tchebi = v.max(axis=1)
        return tchebi
