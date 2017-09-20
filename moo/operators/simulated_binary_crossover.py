import random

import numpy as np

from moo.configuration import Configuration


class SimulatedBinaryCrossover:

    def __init__(self, xl, xu, p_xover=0.9, eta_xover=30):
        self.x_xover = p_xover
        self.eta_xover = eta_xover
        self.xl = xl
        self.xu = xu

    def crossover(self, parent1, parent2):

        n = len(self.xl)

        child1 = np.zeros(n)
        child2 = np.zeros(n)

        if random.random() < self.x_xover:

            for i in range(n):

                if random.random() < 0.5:
                    if abs(parent1[i] - parent2[i]) > Configuration.EPS:

                        if parent1[i] < parent2[i]:
                            y1 = parent1[i]
                            y2 = parent2[i]
                        else:
                            y1 = parent2[i]
                            y2 = parent1[i]

                        yl = self.xl[i]
                        yu = self.xu[i]

                        rand = random.random()
                        beta = 1.0 + (2.0 * (y1 - yl) / (y2 - yl))
                        alpha = 2.0 - np.math.pow(beta, -(self.eta_xover + 1))
                        if rand <= (1.0 / alpha):
                            beta_q = np.math.pow((rand * alpha),
                                                 (1.0 / (self.eta_xover + 1.0)))
                        else:
                            beta_q = np.math.pow((1.0 / (2.0 - rand * alpha)),
                                                 (1.0 / (self.eta_xover + 1.0)))

                        c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                        beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
                        alpha = 2.0 - np.math.pow(beta, -(self.eta_xover + 1.0))
                        if rand <= (1.0 / alpha):
                            beta_q = np.math.pow((rand * alpha),
                                                 (1.0 / (self.eta_xover + 1.0)))
                        else:
                            beta_q = np.math.pow((1.0 / (2.0 - rand * alpha)),
                                                 (1.0 / (self.eta_xover + 1.0)))
                        c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                        if c1 < yl:
                            c1 = yl
                        if c2 < yl:
                            c2 = yl
                        if c1 > yu:
                            c1 = yu
                        if c2 > yu:
                            c2 = yu

                        if random.random() <= 0.5:
                            child1[i] = c2
                            child2[i] = c1
                        else:
                            child1[i] = c1
                            child2[i] = c2

                    else:
                        child1[i] = parent1[i]
                        child2[i] = parent2[i]
                else:
                    child1[i] = parent1[i]
                    child2[i] = parent2[i]
        else:
            child1 = np.array(parent1)
            child2 = np.array(parent2)

        return child1, child2
