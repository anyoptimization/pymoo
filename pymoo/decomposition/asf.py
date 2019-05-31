from pymoo.model.decomposition import Decomposition


class ASF(Decomposition):

    def _do(self, F, weights, **kwargs):
        weights[weights == 0] = 1e-12
        asf = ((F - self.utopian_point) / weights).max(axis=1)
        return asf
