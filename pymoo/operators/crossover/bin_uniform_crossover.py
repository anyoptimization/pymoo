from pymoo.model.crossover import Crossover
from pymoo.rand import random


class BinaryUniformCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def _do(self, p, parents, children, data=None):

        for i in range(p.n_var):

            if random.random() < 0.5:
                children[0,i] = parents[0,i]
                children[1, i] = parents[1, i]
            else:
                children[0, i] = parents[1, i]
                children[1, i] = parents[0, i]

        return children
