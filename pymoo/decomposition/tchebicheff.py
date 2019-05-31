import numpy as np

from pymoo.model.decomposition import Decomposition


class Tchebicheff(Decomposition):

    def _do(self, F, weights, **kwargs):
        v = np.abs(F - self.utopian_point) * weights
        tchebi = np.max(v, axis=1)
        return tchebi
