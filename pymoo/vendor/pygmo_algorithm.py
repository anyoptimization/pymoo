import pygmo as pg
import numpy
from model.algorithm import Algorithm
from vendor.pygmo_problem import create_pygmo_problem


class PygmoAlgorithm(Algorithm):
    def __init__(self, pygmo_algrithm, pop_size=100):
        self.pygmo_algrithm = pygmo_algrithm
        self.pop_size = pop_size

    def solve_(self, problem, _, rnd=None):
        pp = create_pygmo_problem(problem)
        pop = pg.population(pp, size=self.pop_size)
        a = pg.algorithm(self.pygmo_algrithm)
        pop = a.evolve(pop)
        return pop.get_x(), pop.get_f(), numpy.zeros((len(pop),0))

