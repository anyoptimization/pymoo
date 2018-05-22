from pymoo.model.mutation import Mutation
from pymoo.rand import random


class PolynomialMutation(Mutation):
    def __init__(self, eta_mut=20, p_mut=None):
        self.eta_m = eta_mut
        self.p_mut = p_mut

    def _do(self, p, X, Y):

        if self.p_mut is None:
            self.p_mut = 1.0 / p.n_var

        for i in range(X.shape[0]):

            for j in range(X.shape[1]):

                # no mutation for this index
                if random.random() > self.p_mut:
                    Y[i, j] = X[i, j]
                else:
                    y = X[i, j]
                    yl = p.xl[j]
                    yu = p.xu[j]
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)
                    r = random.random()
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
                    Y[i, j] = y

        return Y
