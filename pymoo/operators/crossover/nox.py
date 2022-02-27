from pymoo.core.crossover import Crossover
from pymoo.core.population import Population


class NoCrossover(Crossover):

    def __init__(self):
        super().__init__(1, 1, 0.0)

    def do(self, problem, parents, **kwargs):
        return Population.create(*[p[0] for p in parents])
