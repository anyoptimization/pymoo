from pymoo.model.crossover import Crossover
from pymoo.operators.crossover.util import crossover_mask
from pymoo.rand import random


class UniformCrossover(Crossover):

    def __init__(self, prob=0.5):
        super().__init__(2, 2)
        self.prob = prob

    def _do(self, problem, pop, parents, **kwargs):

        # get the X of parents and count the matings
        X = pop.get("X")[parents.T]
        n_matings = parents.shape[0]

        # random matrix to do the crossover
        M = random.random((n_matings, problem.n_var)) < self.prob

        _X = crossover_mask(X, M)
        return pop.new("X", _X)
