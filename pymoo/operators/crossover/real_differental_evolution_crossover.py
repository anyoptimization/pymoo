from pymoo.model.crossover import Crossover
from pymoo.rand import random

import numpy as np


class DifferentalEvolutionCrossover(Crossover):
    def __init__(self, p_xover=0.5, scale=0.5):
        super().__init__(3, 1)
        self.p_xover = p_xover
        self.scale = scale

    def _do(self, p, parents, children, data=None):

        # create a random binary array with at least one 1
        r = random.random(p.n_var) < 0.5
        if np.sum(r) == 0:
            r[random.randint(0, p.n_var)] = 1

        # do the crossover
        children[0, :] = parents[0, :]
        children[0, r] = parents[0, r] + self.scale * (parents[1, r] - parents[2, r])

        # repair if out of bounds
        larger_than_xu = children[0, :] > p.xu
        children[0, larger_than_xu] = p.xu[larger_than_xu]

        smaller_than_xl = children[0, :] < p.xl
        children[0, smaller_than_xl] = p.xl[smaller_than_xl]

        return children
