import numpy as np

from pymoo.core.fitness import sort_by_fitness
from pymoo.core.solution import merge


class Archive:

    def __init__(self):
        super().__init__()
        self.sols = None

    def add(self, sols):
        self.sols = merge(self.sols, sols)
        self.reduce()
        return self.sols

    def solution(self):
        if len(self.sols) > 0:
            return self.sols[0]

    def solutions(self):
        return self.sols

    def reduce(self):
        pass


class Optimum(Archive):

    def reduce(self):
        self.sols = sort_by_fitness(self.sols)[[0]]
