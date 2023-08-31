import numpy as np
from pymoo import PYMOO_PRNG
from pymoo.core.crossover import Crossover
from pymoo.core.population import Population


class NoCrossover(Crossover):

    def __init__(self):
        super().__init__(1, 1, 0.0)

    def do(self, problem, pop, **kwargs):
        return Population.create(*[PYMOO_PRNG.choice(parents) for parents in pop])
