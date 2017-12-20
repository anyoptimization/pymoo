import numpy as np

from pymoo.model import random


class TournamentSelection:
    def __init__(self,
                 func=None,
                 pressure=2):

        self.pressure = pressure

        self.func = func
        if self.func is None:
            self.func = self.select_by_min_index

    def _initialize(self, pop):
        self.pop = pop
        self.perm = random.perm(self.pop.size())
        self.counter = 0

    def next(self, n_selected):

        selected = np.zeros(n_selected, dtype=np.int)

        for i in range(n_selected):

            if self.counter + self.pressure >= self.pop.size():
                self.perm = random.perm(self.pop.size())
                self.counter = 0
            selected[i] = self.func(self.pop, self.perm[self.counter:self.counter + self.pressure])
            self.counter = self.counter + self.pressure

        return selected

    def select_by_min_index(self, pop, indices):
        return np.min(indices)
