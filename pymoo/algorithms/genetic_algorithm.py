import math

import numpy as np
from scipy.spatial.distance import cdist

from pymoo.model.algorithm import Algorithm
from pymoo.model.individual import Individual
from pymoo.model.population import Population
from pymoo.rand import random


class GeneticAlgorithm(Algorithm):

    def __init__(self,
                 pop_size,
                 sampling,
                 selection,
                 crossover,
                 mutation,
                 survival,
                 n_offsprings=None,
                 eliminate_duplicates=False,
                 repair=None,
                 individual=Individual(),
                 **kwargs
                 ):

        super().__init__(**kwargs)

        # population size of the genetic algorithm
        self.pop_size = pop_size

        # initial sampling method: object, 2d array, or population (already evaluated)
        self.sampling = sampling

        # the method to be used to select parents for recombination
        self.selection = selection

        # method to do the crossover
        self.crossover = crossover

        # method for doing the mutation
        self.mutation = mutation

        # function to repair an offspring after mutation if necessary
        self.repair = repair

        # survival selection
        self.survival = survival

        # number of offsprings to generate through recombination
        self.n_offsprings = n_offsprings

        # a function that returns the indices of duplicates
        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = default_is_duplicate
            else:
                self.eliminate_duplicates = None
        elif callable(eliminate_duplicates):
            self.eliminate_duplicates = eliminate_duplicates
        else:
            raise Exception("eliminate_duplicates can be either a function or bool for default elimination function.")

        # the object to be used to represent an individual - either individual or derived class
        self.individual = individual

        # if the number of offspring is not set - equal to population size
        if self.n_offsprings is None:
            self.n_offsprings = pop_size

        # other run specific data updated whenever solve is called - to share them in all methods
        self.n_gen = None
        self.pop = None
        self.off = None

    def _solve(self, problem, termination):

        # generation counter
        self.n_gen = 1

        # initialize the first population and evaluate it
        self.pop = self._initialize()
        self._each_iteration(self, first=True)

        # while termination criterium not fulfilled
        while termination.do_continue(self):
            self.n_gen += 1

            # do the next iteration
            self.pop = self._next(self.pop)

            # execute the callback function in the end of each generation
            self._each_iteration(self)

        self._finalize()

        return self.pop

    def _initialize(self):
        # ! get the initial population - different ways are possible

        # provide a whole population object - (individuals might be already evaluated)
        if isinstance(self.sampling, Population):
            pop = self.sampling
        else:
            pop = Population(0, individual=self.individual)
            if isinstance(self.sampling, np.ndarray):
                pop = pop.new("X", self.sampling)
            else:
                pop = self.sampling.sample(self.problem, pop, self.pop_size, algorithm=self)

        # in case the initial population was not evaluated
        if np.any(pop.collect(lambda ind: ind.F is None, as_numpy_array=True)):

            # repair first in case it is necessary
            if self.repair:
                pop = self.repair.do(self.problem, pop, algorithm=self)

            # then evaluate using the objective function
            self.evaluator.eval(self.problem, pop, algorithm=self)

        # that call is a dummy survival to set attributes that are necessary for the mating selection
        if self.survival:
            pop = self.survival.do(self.problem, pop, self.pop_size, algorithm=self)

        return pop

    def _next(self, pop):

        # do the mating using the current population
        self.off = self._mating(pop)

        # evaluate the offspring
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        # merge the offsprings with the current population
        pop = pop.merge(self.off)

        # the do survival selection
        pop = self.survival.do(self.problem, pop, self.pop_size, algorithm=self)

        return pop

    def _mating(self, pop):

        # the population object to be used
        off = pop.new()

        # mating counter - counts how often the mating needs to be done to fill up n_offsprings
        n_matings = 0

        # iterate until enough offsprings are created
        while len(off) < self.n_offsprings:

            # how many parents need to be select for the mating - depending on number of offsprings remaining
            n_select = math.ceil((self.n_offsprings - len(off)) / self.crossover.n_offsprings)

            # select the parents for the mating - just an index array
            parents = self.selection.do(pop, n_select, self.crossover.n_parents, algorithm=self)

            # do the crossover using the parents index and the population - additional data provided if necessary
            _off = self.crossover.do(self.problem, pop, parents, algorithm=self)

            # do the mutation on the offsprings created through crossover
            _off = self.mutation.do(self.problem, _off, algorithm=self)

            # repair the individuals if necessary
            if self.repair:
                _off = self.repair.do(self.problem, _off, algorithm=self)

            if self.eliminate_duplicates:
                is_duplicate = self.eliminate_duplicates(_off, pop, off, algorithm=self)
                _off = _off[np.logical_not(is_duplicate)]

            # if more offsprings than necessary - truncate them
            if len(_off) > self.n_offsprings - len(off):
                I = random.perm(self.n_offsprings - len(off))
                _off = _off[I]

            # add to the offsprings and increase the mating counter
            off = off.merge(_off)
            n_matings += 1

            # if no new offsprings can be generated within 100 trails -> return the current result
            if n_matings > 100:
                print(
                    "WARNING: Recombination could not produce new offsprings which are not already in the population!")
                break

        return off

    def _finalize(self):
        pass


def default_is_duplicate(pop, *other, epsilon=1e-20, **kwargs):
    if len(other) == 0:
        return np.full(len(pop), False)

    X = pop.get("X")

    # value to finally return
    is_duplicate = np.full(len(pop), False)

    # check for duplicates in pop itself
    D = cdist(X, X)
    D[np.triu_indices(len(pop))] = np.inf
    is_duplicate = np.logical_or(is_duplicate, np.any(D < epsilon, axis=1))

    # check for duplicates to all others
    _X = other[0].get("X")
    for o in other[1:]:
        if len(o) > 0:
            _X = np.concatenate([_X, o.get("X")])

    is_duplicate = np.logical_or(is_duplicate, np.any(cdist(X, _X) < epsilon, axis=1))

    return is_duplicate
