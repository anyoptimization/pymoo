from pymoo.model.crossover import Crossover
from pymoo.rand import random

import numpy as np

from pymoo.util.misc import repair


class DifferentalEvolutionCrossover(Crossover):
    def __init__(self, p_xover=0.5, scale=0.3, repair=True):
        super().__init__(3, 1)
        self.p_xover = p_xover
        self.scale = scale
        self.repair = repair

    def _do(self, p, parents, children, data=None):

        # create a random binary array with at least one 1
        r = random.random(p.n_var) <= self.p_xover
        if np.sum(r) == 0:
            r[random.randint(0, p.n_var)] = 1

        # do the crossover
        children[0, :] = parents[0, :]
        children[0, r] = parents[0, r] + self.scale * (parents[1, r] - parents[2, r])

        # repair if out of bounds
        if self.repair:
            children = repair(children, p.xl, p.xu)

        return children
