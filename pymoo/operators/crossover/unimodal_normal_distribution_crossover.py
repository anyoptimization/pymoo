import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.operators.crossover.util import crossover_mask
from pymoo.rand import random


class UNDX(Crossover):

    def __init__(self, n_parents):
        super().__init__(n_parents, 1)

    def _do(self, problem, pop, parents, **kwargs):

        # get the X of parents and count the matings
        X = pop.get("X")[parents.T]
        _, n_matings, n_var = X.shape

        raise Exception("NOT implemented yet.")


        _X = None
        return pop.new("X", _X)
pre