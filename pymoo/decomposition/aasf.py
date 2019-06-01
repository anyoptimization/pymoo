import numpy as np

from pymoo.decomposition.asf import ASF


class AASF(ASF):

    def __init__(self, eps=1e-10, _type="auto", rho=None, beta=None, **kwargs) -> None:
        super().__init__(eps, _type, **kwargs)
        if rho is None and beta is None:
            raise Exception("Either provide rho or beta!")
        elif rho is None:
            self.rho = calc_rho(beta)
        else:
            self.rho = rho

    def _do(self, F, weights, **kwargs):
        asf = super()._do(F, weights, **kwargs)
        augment = ((F - self.utopian_point) / weights).sum(axis=1)
        return asf + self.rho * augment


def calc_rho(beta):
    return 1 / (1 - np.tan(beta / 360 * 2 * np.pi)) - 1
