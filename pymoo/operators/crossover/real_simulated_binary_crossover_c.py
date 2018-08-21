import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.rand import random


class SimulatedBinaryCrossover(Crossover):
    def __init__(self, prob_cross, eta_cross):
        super().__init__(2, 2)
        self.prob_cross = float(prob_cross)
        self.eta_cross = float(eta_cross)

    def _do(self, p, parents, children, **kwargs):

        n_children = 0

        for k in range(parents.shape[0]):

            if random.random() <= self.prob_cross:

                for i in range(p.n_var):

                    if random.random() <= 0.5:

                        if np.abs(parents[k, 0, i] - parents[k, 1, i]) > 1.0e-14:

                            if parents[k, 0, i] < parents[k, 1, i]:
                                y1 = parents[k, 0, i]
                                y2 = parents[k, 1, i]
                            else:
                                y1 = parents[k, 1, i]
                                y2 = parents[k, 0, i]

                            yl = p.xl[i]
                            yu = p.xu[i]
                            rand = random.random()
                            beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
                            alpha = 2.0 - pow(beta, -(self.eta_cross + 1.0))
                            if rand <= (1.0 / alpha):
                                betaq = pow((rand * alpha), (1.0 / (self.eta_cross + 1.0)))

                            else:
                                betaq = pow((1.0 / (2.0 - rand * alpha)), (1.0 / (self.eta_cross + 1.0)))

                            c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

                            beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
                            alpha = 2.0 - pow(beta, -(self.eta_cross + 1.0))

                            if rand <= (1.0 / alpha):
                                betaq = pow((rand * alpha), (1.0 / (self.eta_cross + 1.0)))
                            else:
                                betaq = pow((1.0 / (2.0 - rand * alpha)), (1.0 / (self.eta_cross + 1.0)))

                            c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                            if c1 < yl:
                                c1 = yl
                            if c2 < yl:
                                c2 = yl
                            if c1 > yu:
                                c1 = yu
                            if c2 > yu:
                                c2 = yu

                            if random.random() <= 0.5:
                                children[n_children, i] = c2
                                children[n_children + 1, i] = c1
                            else:
                                children[n_children, i] = c1
                                children[n_children + 1, i] = c2

                        else:
                            children[n_children, i] = parents[k, 0, i]
                            children[n_children + 1, i] = parents[k, 1, i]

                    else:
                        children[n_children, i] = parents[k, 0, i]
                        children[n_children + 1, i] = parents[k, 1, i]

            else:
                children[n_children, :] = parents[k, 0, :]
                children[n_children + 1, :] = parents[k, 1, :]

            n_children += self.n_children
