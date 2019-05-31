import numpy as np

from pymoo.decomposition.asf import ASF


class AASF(ASF):

    def _do(self, F, weights,
            rho=None, beta=None, **kwargs):

        if rho is None and beta is None:
            raise Exception("Either provide rho or beta!")
        elif rho is None:

            rho = 1 / (1-np.tan(beta / 360 * 2 * np.pi)) - 1

        asf = super()._do(F, weights, **kwargs)
        augment = ((F - self.utopian_point) / weights).sum(axis=1)

        return asf + rho * augment
