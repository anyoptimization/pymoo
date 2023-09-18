import numpy as np
import pymoo
from pymoo.core.crossover import Crossover
from pymoo.core.population import Population


class NoCrossover(Crossover):

    def __init__(self):
        super().__init__(1, 1, 0.0)

    def do(self, problem, pop, **kwargs):
        return Population.create(*[pymoo.PymooPRNG().choice(parents) for parents in pop])
