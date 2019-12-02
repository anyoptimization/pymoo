import math

import numpy as np

from pymoo.model.algorithm import Algorithm
from pymoo.model.duplicate import DefaultDuplicateElimination
from pymoo.model.individual import Individual
from pymoo.model.population import Population


class GeneticAlgorithm(Algorithm):

    def __init__(self,
                 pop_size,
                 sampling,
                 selection,
                 crossover,
                 mutation,
                 survival,
                 n_offsprings=None,
                 eliminate_duplicates=DefaultDuplicateElimination(),
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

        # set the duplicate detection class - a boolean value chooses the default duplicate detection
        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = None
        else:
            self.eliminate_duplicates = eliminate_duplicates

        # the object to be used to represent an individual - either individual or derived class
        self.individual = individual

        # if the number of offspring is not set - equal to population size
        if self.n_offsprings is None:
            self.n_offsprings = pop_size

        # other run specific data updated whenever solve is called - to share them in all algorithms
        self.n_gen = None
        self.pop = None
        self.off = None

        # this can be used to store additional data in submodules e.g. survival, recombination and so on
        self.data = {}

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
                pop = self.sampling.do(self.problem, self.pop_size, pop=pop, algorithm=self)

        # repair all solutions that are not already evaluated
        if self.repair:
            I = [k for k in range(len(pop)) if pop[k].F is None]
            pop = self.repair.do(self.problem, pop[I], algorithm=self)

        # then evaluate using the objective function
        self.evaluator.eval(self.problem, pop, algorithm=self)

        # that call is a dummy survival to set attributes that are necessary for the mating selection
        if self.survival:
            pop = self.survival.do(self.problem, pop, len(pop), algorithm=self)

        self.pop = pop

    def _next(self):

        # do the mating using the current population
        self.off = self._mating(self.pop, n_max_iterations=100)

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(self.off) == 0:
            self.termination.force_termination = True
            return
        # if not the desired number of offspring could be created
        elif len(self.off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        # evaluate the offspring
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        # merge the offsprings with the current population
        self.pop = self.pop.merge(self.off)

        # the do survival selection
        self.pop = self.survival.do(self.problem, self.pop, self.pop_size, algorithm=self)

    def _mating(self, pop, n_max_iterations=100):

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

            if self.eliminate_duplicates is not None:
                _off = self.eliminate_duplicates.do(_off, pop, off)

            # if more offsprings than necessary - truncate them randomly
            if len(off) + len(_off) > self.n_offsprings:
                n_remaining = self.n_offsprings - len(off)
                I = np.random.permutation(len(_off))[:n_remaining]
                _off = _off[I]

            # add to the offsprings and increase the mating counter
            off = off.merge(_off)
            n_matings += 1

            # if no new offsprings can be generated within a pre-specified number of generations
            if n_matings > n_max_iterations:
                break

        return off

    def _finalize(self):
        pass
