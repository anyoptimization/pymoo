from pymoo.model.mutation import Mutation
from pymoo.rand import random


class PolynomialMutation(Mutation):
    def __init__(self, eta_mut, prob_mut=None):
        self.eta_mut = float(eta_mut)
        if prob_mut is not None:
            self.prob_mut = float(prob_mut)
        else:
            self.prob_mut = None

    def _do(self, p, X, Y, **kwargs):

        if self.prob_mut is None:
            self.prob_mut = 1.0 / p.n_var

        for i in range(X.shape[0]):

            for j in range(X.shape[1]):

                rnd = random.random()
                if rnd <= self.prob_mut:

                    y = X[i, j]
                    yl = p.xl[j]
                    yu = p.xu[j]

                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)

                    mut_pow = 1.0 / (self.eta_mut + 1.0)

                    rnd = random.random()
                    if rnd <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (pow(xy, (self.eta_mut + 1.0)))
                        deltaq = pow(val, mut_pow) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (pow(xy, (self.eta_mut + 1.0)))
                        deltaq = 1.0 - (pow(val, mut_pow))

                    y = y + deltaq * (yu - yl)

                    if y < yl:
                        y = yl

                    if y > yu:
                        y = yu

                    Y[i, j] = y

                else:
                    Y[i, j] = X[i, j]
