from pymoo.core.decomposition import Decomposition


class ASF(Decomposition):

    def _do(self, F, weights, weight_0=1e-10, **kwargs):
        _weights = weights.astype(float)
        _weights[weights == 0] = weight_0
        asf = ((F - self.utopian_point) / _weights).max(axis=1)
        return asf
