from pymoo.model.selection import Selection
from pymoo.rand import random


class RandomSelection(Selection):
    def __init__(self):
        self.perm = None
        self.counter = None
        self.pop = None

    def set_population(self, pop, data):
        self.pop = pop

    def next(self, n_selected):
        if self.perm is None or self.counter + n_selected >= self.pop.size():
            self.perm = random.perm(self.pop.size())
            self.counter = 0

        selected = self.perm[self.counter:self.counter+n_selected]
        self.counter = self.counter + n_selected

        return selected
