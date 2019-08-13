import autograd.numpy as anp

from pymoo.model.decomposition import Decomposition


class Tchebicheff(Decomposition):

    def _do(self, F, weights, **kwargs):
        v = anp.abs(F - self.utopian_point) * weights
        tchebi = v.max(axis=1)
        return tchebi
