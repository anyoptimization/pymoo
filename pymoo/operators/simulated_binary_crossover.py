import random

import numpy as np

from configuration import Configuration
from rand.default_random_generator import DefaultRandomGenerator


class SimulatedBinaryCrossover:
    def __init__(self, eta_xover=15, p_xover=0.9):
        self.p_xover = p_xover
        self.eta_c = eta_xover
        self.n_parents = 2
        self.n_children = 2

    def crossover(self, parent1, parent2, xl, xu, rnd=DefaultRandomGenerator()):

        n = len(xl)

        child1 = np.zeros(n)
        child2 = np.zeros(n)

        if rnd.random() <= self.p_xover:

            for i in range(n):

                if rnd.random() <= 0.5:

                    if abs(parent1[i] - parent2[i]) > Configuration.EPS:

                        if parent1[i] < parent2[i]:
                            y1 = parent1[i]
                            y2 = parent2[i]
                        else:
                            y1 = parent2[i]
                            y2 = parent1[i]

                        yl = xl[i]
                        yu = xu[i]
                        rand = rnd.random()
                        beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.eta_c + 1.0))
                        if rand <= (1.0 / alpha):
                            betaq = pow((rand * alpha), (1.0 / (self.eta_c + 1.0)))

                        else:
                            betaq = pow((1.0 / (2.0 - rand * alpha)), (1.0 / (self.eta_c + 1.0)))

                        c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                        beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.eta_c + 1.0))
                        if rand <= (1.0 / alpha):
                            betaq = pow((rand * alpha), (1.0 / (self.eta_c + 1.0)))
                        else:
                            betaq = pow((1.0 / (2.0 - rand * alpha)), (1.0 / (self.eta_c + 1.0)))

                        c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                        if c1 < yl:
                            c1 = yl
                        if c2 < yl:
                            c2 = yl
                        if c1 > yu:
                            c1 = yu
                        if c2 > yu:
                            c2 = yu
                        if rnd.random() <= 0.5:
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
            child1 = np.array(parent1, copy=True)
            child2 = np.array(parent2, copy=True)

        return [child1, child2]
