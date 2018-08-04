from pymoo.model.crossover import Crossover
from pymoo.rand import random


class BinaryUniformCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def _do(self, p, parents, children, **kwargs):

        # number of parents
        n_parents = parents.shape[0]

        # random matrix to do the crossover
        M = random.random(n_parents, p.n_var)
        smaller, larger = M < 0.5, M > 0.5

        # first possibility where first parent 0 is copied
        children[:n_parents][smaller] = parents[:, 0, :][smaller]
        children[:n_parents][larger] = parents[:, 1, :][larger]

        # now flip the order of parents with the same random array and write the second half of children
        children[n_parents:][smaller] = parents[:, 1, :][smaller]
        children[n_parents:][larger] = parents[:, 0, :][larger]
