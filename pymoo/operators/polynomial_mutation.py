import numpy as np

from rand.default_random_generator import DefaultRandomGenerator


class PolynomialMutation:
    def __init__(self, eta_mut=20, p_mut=None):
        self.eta_m = eta_mut
        self.p_mut = p_mut

    def mutate(self, x, xl, xu, rnd=DefaultRandomGenerator()):

        if self.p_mut is None:
            self.p_mut = 1.0 / len(x)

        res = np.array(x, copy=True)

        for j in range(len(x)):

            if rnd.random() <= self.p_mut:

                y = x[j]
                yl = xl[j]
                yu = xu[j]
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                r = rnd.random()
                mut_pow = 1.0 / (self.eta_m + 1.0)
                if r <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * r + (1.0 - 2.0 * r) * (pow(xy, (self.eta_m + 1.0)))
                    deltaq = pow(val, mut_pow) - 1.0

                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (pow(xy, (self.eta_m + 1.0)))
                    deltaq = 1.0 - (pow(val, mut_pow))

                y = y + deltaq * (yu - yl)
                if y < yl:
                    y = yl
                if y > yu:
                    y = yu
                res[j] = y

        return res
