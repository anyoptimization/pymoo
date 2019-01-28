import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.rand import random


class BinaryUniformCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def _do(self, problem, pop, parents, **kwargs):

        # number of parents
        n_matings = parents.shape[0]
        off = np.full((n_matings * self.n_offsprings, problem.n_var), np.inf, dtype=problem.type_var)

        X = pop.get("X")[parents.T]

        # random matrix to do the crossover
        M = random.random((n_matings, problem.n_var))
        smaller, larger = M < 0.5, M > 0.5

        # first possibility where first parent 0 is copied
        off[:n_matings][smaller] = X[0, :, :][smaller]
        off[:n_matings][larger] = X[1, :, :][larger]

        # now flip the order of parents with the same random array and write the second half of off
        off[n_matings:][smaller] = X[1, :, :][smaller]
        off[n_matings:][larger] = X[0, :, :][larger]

        return pop.new("X", off.astype(problem.type_var))
