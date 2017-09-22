import numpy as np


class PolynomialMutation:
    def __init__(self, eta_mut=20, p_mut=None):
        self.eta_mut = eta_mut
        self.p_mut = p_mut

    def mutate(self, x, xl, xu):

        if self.p_mut is None:
            self.p_mut = 1.0 / len(x)

        res = np.zeros(len(x))
        for j in range(len(x)):
            if np.random.random() < self.p_mut:
                y = x[j]
                yl = xl[j]
                yu = xu[j]
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                rnd = np.random.random()
                mut_pow = 1.0 / (self.eta_mut + 1.0)
                if rnd <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (
                        np.math.pow(xy, (self.eta_mut + 1.0)))
                    delta_q = np.math.pow(val, mut_pow) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (
                        np.math.pow(xy, (self.eta_mut + 1.0)))
                    delta_q = 1.0 - (np.math.pow(val, mut_pow))

                y = y + delta_q * (yu - yl)
                if y < yl:
                    y = yl
                if y > yu:
                    y = yu
                res[j] = y

            else:
                res[j] = x[j]

        return res
