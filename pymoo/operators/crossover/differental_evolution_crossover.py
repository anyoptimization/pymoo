from pymoo.model.crossover import Crossover
from pymoo.rand import random


class DifferentialEvolutionCrossover(Crossover):

    def __init__(self, weight=0.8, dither=None, jitter=False):
        super().__init__(3, 1)
        self.weight = weight
        self.dither = dither
        self.jitter = jitter

    def _do(self, problem, pop, parents, **kwargs):
        X = pop.get("X")[parents.T]

        if self.dither == "vector":
            weight = (self.weight + random.random(len(parents)) * (1 - self.weight))[:, None]
        elif self.dither == "scalar":
            weight = self.weight + random.random() * (1 - self.weight)
        else:
            weight = self.weight

        # http://www.cs.ndsu.nodak.edu/~siludwig/Publish/papers/SSCI20141.pdf
        if self.jitter:
            gamma = 0.0001
            weight = (self.weight * (1 + gamma * (random.random(len(parents)) - 0.5)))[:, None]

        _X = X[0] + weight * (X[1] - X[2])
        return pop.new("X", _X)
