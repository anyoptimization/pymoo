import numpy as np


class TournamentSelection:
    def __init__(self):
        pass

    def initialize(self,
                   pop,
                   func=None,
                   pressure=2):
        self.pop = pop
        self.pressure = pressure
        self.perm = np.random.permutation(self.pop.size())
        self.counter = 0

        self.func = func
        if self.func is None:
            self.func = self.select_by_min_index

    def next(self, n_selected):

        selected = np.zeros(n_selected, dtype=np.int)

        for i in range(n_selected):

            if self.counter + self.pressure >= self.pop.size():
                self.perm = np.random.permutation(self.pop.size())
                self.counter = 0
            selected[i] = self.func(self.pop, self.perm[self.counter:self.counter + self.pressure])
            self.counter = self.counter + self.pressure

        return selected


    def select_by_min_index(self, pop, indices):
        return np.min(indices)
